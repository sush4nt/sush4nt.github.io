---
title: "All Tree Models — v1 (Comprehensive Intuition Guide)"
date: 2026-07-22T00:00:00+05:30
draft: true
tags: ["tree-models", "ml-fundamentals", "interview-prep"]
summary: "A walk through every major tree-based model, where each section ends with the specific weakness that motivated the next algorithm."
---

# All Tree Models — v1 (Comprehensive Intuition Guide)

> **How to read this document**: each section covers one algorithm and ends with "**The weakness that motivates the next model**." That sentence is the thread connecting the whole document — every algorithm below exists because of a specific, nameable failure of the one before it. If you can recite that chain in an interview, you've demonstrated the kind of systems-level understanding a senior interviewer is actually probing for (not "do you know the formula" but "do you know *why this formula exists*").

## Roadmap — the full chain (Interview-Ready Conceptual Backbone)

This table is the **conceptual backbone** of tree-based modeling — each algorithm solves a specific, concrete problem of its predecessor. For interviews, this is your anchor: if you can articulate the progression and *why* each step exists, you're demonstrating the systems thinking that separates senior engineers from those who just memorize formulas.

| # | Model | Core Mechanism | The Prior Model's Weakness | Why This Matters |
|---|---|---|---|---|
| **1** | **Decision Tree (CART)** | Greedy recursive binary splitting via impurity reduction (Gini/entropy for classification, variance for regression). Stops when a criterion is met (depth, samples per leaf, impurity gain). | — (Foundation algorithm) | Provides the basic vocabulary: splits, leaves, predictions. But single trees are unstable — small data perturbations → completely different tree structure. High variance. |
| **2** | **Random Forest** | Bagging (bootstrap samples) + forced feature subsampling at every node. Train $B$ independent trees in parallel; average predictions (regression) or vote (classification). | **Single tree instability**: one tree achieves low bias but high variance on unseen data. Overfits easily; poor generalization. | Reduces variance without increasing bias (decorrelated trees don't reduce bias individually, but averaging many uncorrelated errors cancels out). Key insight: averaging is powerful if predictors are independent. |
| **3** | **AdaBoost** | Sequential boosting: reweight misclassified samples exponentially; train weak learners (usually stumps) on reweighted data. Combine with weighted voting (higher weight = stronger tree). | **Bias problem**: RF averages independent trees, which doesn't reduce the shared bias they all have. High-error regions aren't handled better than low-error regions. Boosting fixes *bias*, not just variance. | Corrective learning: each new tree explicitly targets the hardest samples from prior trees. Exponential margin concept ensures high confidence on most samples, but outliers stay hard — no built-in regularization. |
| **4** | **Gradient Boosting (GBM)** | Generalize boosting via **gradient descent in function space**: at each step, compute pseudo-residuals ($-\nabla L$), fit a tree to them, update ensemble by stepping along gradient. Works with *any* differentiable loss. | **Loss inflexibility**: AdaBoost is locked to exponential loss for classification. No principled way to handle other objectives (AUC, quantiles, custom business metrics). One-size-fits-all, not adaptive. | Unified framework: same algorithm for classification, regression, ranking, or custom losses. Outlier robustness via Huber loss. But no regularization — overfitting on noisy data or small validation sets. |
| **5** | **XGBoost** | **Regularized objective** ($\gamma T + \lambda w^2$ terms); **second-order Taylor approximation** (Hessian, not just gradient) for split scoring; **histogram-based splits** and default directions for missing values. | **GBM's loose regularization**: only learning rate + early stopping control overfitting; no complexity penalty in the objective. Slow on large data (sorting cost). No native missing value handling. | Production-grade: explicit regularization prevents overfitting even with limited validation data. Second-order splits are faster to converge (fewer iterations). Native missing handling (learned default direction per feature). Industry standard for high-accuracy applications. |
| **6** | **LightGBM** | **Leaf-wise tree growth** (best-first, not level-wise); **GOSS sampling** (keep high-gradient, sample low-gradient samples); **EFB bundling** (merge mutually exclusive features); **sparse-aware histograms**. | **XGBoost's scale bottleneck**: level-wise growth creates balanced, memory-heavy trees. Full data per iteration. On 100M samples with 10K features, even with histograms, memory and time dominate. Doesn't exploit sparsity or feature redundancy. | Extreme scale: 5–20× faster than XGBoost on large, sparse, high-dimensional data. Leaf-wise growth converges in fewer iterations (each split targets highest-error region). GOSS removes 70% of low-error samples without losing signal. EFB reduces one-hot from 10K to 100 features. |
| **7** | **CatBoost** | **Ordered Target Statistics (OTS)**: encode categories as mean target value, computed in an ordered/sequential way to prevent target leakage. **Ordered boosting**: trees only see statistics from "older" samples, preventing direct optimization on category-to-target association. | **Categorical leakage**: XGBoost/LightGBM require manual encoding (one-hot, ordinal, or target encoding). Any encoding risks target leakage — the model can learn spurious category patterns that don't generalize. Especially bad for rare categories. | Categorical robustness: natively handles high-cardinality categoricals without manual engineering or leakage. No need for one-hot explosion. Better calibration and generalization on categorical-heavy datasets (e.g., ad tech, e-commerce). |

### How to use this roadmap in an interview

1. **Show the progression, not the details**: "The evolution is variance-reduction (RF) → bias-reduction (AdaBoost) → generalization (GBM) → robustness (XGBoost) → scale (LightGBM) → categorical handling (CatBoost)."
2. **Name the specific weakness each fixes**: Interviewer asks "Why XGBoost?" → "GBM has no built-in regularization and is slow on large data. XGBoost adds explicit L1/L2 terms in the objective *and* uses Hessian-based splits so you need fewer trees."
3. **Tie to production context**: "In practice, if your data is < 1M rows and categorical-light, XGBoost is solid. If you have 100M rows with one-hot features, LightGBM. If your data is mostly high-cardinality categoricals, CatBoost avoids encoding headaches."
4. **Emphasize tradeoffs, not "best"**: "Each algorithm makes a different bet. RF doesn't need tuning but is high-variance. XGBoost needs tuning but gives better accuracy. LightGBM is faster but overfits more easily. CatBoost is robust but slower on non-categorical data."

## Running Example Datasets (used identically across every model in this document)

Using the **same two toy datasets** throughout means you can directly compare what changes about how each algorithm treats the *same* data — which is the fastest way to build real intuition.

### Classification dataset — Loan Default Prediction (binary)

| ID | Income ($k) | CreditScore | DebtRatio | **Default** |
|----|---|---|---|---|
| 1 | 45 | 580 | 0.45 | Yes |
| 2 | 60 | 620 | 0.35 | Yes |
| 3 | 80 | 700 | 0.20 | No |
| 4 | 65 | 645 | 0.30 | **Yes** *(noisy/borderline point)* |
| 5 | 90 | 750 | 0.15 | No |
| 6 | 55 | 600 | 0.40 | Yes |
| 7 | 100 | 780 | 0.10 | No |
| 8 | 70 | 690 | 0.25 | No |
| 9 | 40 | 560 | 0.50 | Yes |
| 10 | 85 | 720 | 0.18 | No |

Note row 4 is deliberately non-separable by CreditScore alone — real data is noisy, and this is what forces every algorithm in this document to make actual tradeoffs instead of trivially achieving 100% purity in one split.

### Regression dataset — House Price Prediction

| ID | SqFt (×100) | Age (yrs) | Bedrooms | **Price ($k)** |
|----|---|---|---|---|
| 1 | 8 | 5 | 2 | 180 |
| 2 | 12 | 2 | 3 | 250 |
| 3 | 20 | 10 | 4 | 320 |
| 4 | 9 | 20 | 2 | 150 |
| 5 | 15 | 8 | 3 | 270 |
| 6 | 25 | 1 | 5 | 400 |
| 7 | 11 | 15 | 2 | 165 |
| 8 | 18 | 3 | 4 | 310 |
| 9 | 7 | 25 | 1 | 120 |
| 10 | 22 | 6 | 4 | 350 |

---

# Section 1 — Decision Trees (CART)

## 1.1 The core idea

A decision tree predicts by **recursively partitioning the feature space into axis-aligned rectangles**, and assigning a constant prediction to each rectangle (leaf): majority class for classification, mean target value for regression. Every split is chosen **greedily** — at each node, pick the single feature + threshold that most reduces impurity *right now*, with no lookahead into how that choice affects splits further down the tree.

This greediness is the single most important fact to internalize about decision trees, because it explains almost every property discussed below: why training is fast (no combinatorial search over tree structures), why trees are *not* globally optimal (a provably NP-hard problem — Hyafil & Rivest, 1976 — so greedy is the only tractable approach), and why trees are unstable (a small change in data can flip which split looks best at the root, cascading into a completely different tree).

## 1.2 The objective function — what is actually being minimized

Formally, CART seeks the tree structure $T$ minimizing total loss across all leaves, with leaf predictions $c_m$ set optimally for each leaf:

$$
\hat T = \arg\min_{T} \sum_{m=1}^{|T|} \sum_{i \in R_m} L(y_i, c_m)
$$

where $R_m$ is the region (leaf) and $L$ is the loss function appropriate to the task:

- **Classification**: $L$ is 0-1 loss; the optimal $c_m$ for a leaf is the **majority class**.
- **Regression**: $L$ is squared error; the optimal $c_m$ for a leaf is the **mean of $y_i$ in that leaf** (this is a calculus fact: the value minimizing $\sum (y_i-c)^2$ is $\bar y$).

Because finding the globally optimal $T$ is intractable, CART substitutes a **greedy surrogate objective at each node**: instead of directly minimizing $L$ (0-1 loss / SSE) — which is non-differentiable or too coarse to compare candidate splits meaningfully — it minimizes a smoother **impurity criterion** (Gini, entropy, or variance — see Document 1) as a proxy. This is an important distinction: *the split-selection criterion and the leaf-prediction loss are not identical functions*, they're aligned proxies. Gini/entropy approximate 0-1 loss; variance reduction *is* SSE reduction exactly (which is the one case where proxy = true objective).

## 1.3 The algorithm — 7-step template

| Step | What happens | How it serves the objective |
|---|---|---|
| **1. Initialize** | Root node = entire training set | Starting point of the recursive partition |
| **2. Check stopping criteria** | Is node pure? At `max_depth`? Below `min_samples_split`? | Prevents infinite recursion; first line of generalization control |
| **3. Generate candidate splits** | For each feature, sort its values; candidate thresholds = midpoints between consecutive sorted values (continuous) or subset partitions (categorical) | Defines the search space at this node — this is what makes split-finding tractable: $O(n)$ candidates per feature instead of infinite real-valued thresholds |
| **4. Score every candidate split** | Compute weighted impurity of the two children for each (feature, threshold) pair | This *is* the local optimization — each candidate is scored against the same proxy objective (Gini/entropy/variance) |
| **5. Select the best split** | Pick $(feature^*, threshold^*) = \arg\max_{\text{split}} \Delta\text{impurity}$ | Greedy local maximization step — the literal "decision" the tree makes |
| **6. Partition the data** | Route samples with $x_{feature^*} \leq threshold^*$ to the left child, else right | Physically creates the two new nodes to recurse into |
| **7. Recurse** | Repeat steps 2–6 independently on left and right children | Builds the tree depth-first or breadth-first; each subtree is solved as an independent smaller instance of the same problem |

*(Some implementations add an 8th step: **cost-complexity pruning** after full growth — covered in 1.6 below, since it operates on the already-built tree rather than during growth.)*

## 1.4 Worked example — Classification split (Loan Default)

**Root node** (all 10 samples): 5 Yes, 5 No → $p_{Yes}=0.5$

$$
Gini_{root} = 1-(0.5^2+0.5^2) = 0.5 \quad\text{(maximum possible impurity for binary)}
$$

**Candidate split: CreditScore ≤ 667.5** (midpoint between 645 and 690)

- Left child (CreditScore ≤ 667.5): IDs {1,2,4,6,9} → labels {Yes,Yes,Yes,Yes,Yes} = 5 Yes, 0 No → $Gini_L = 1-(1^2+0^2)=0$
- Right child (CreditScore > 667.5): IDs {3,5,7,8,10} → all No → $Gini_R = 0$

$$
Gini_{weighted} = \tfrac{5}{10}(0) + \tfrac{5}{10}(0) = 0
$$
$$
\Delta Gini = 0.5 - 0 = 0.5 \quad\text{(perfect split — maximum possible gain)}
$$

Row 4 (Income 65, CreditScore 645, Default=Yes) was designed to be the noisy point, but it happens to land *exactly* on the "Yes" side of this threshold, so this particular feature/threshold combination still achieves a clean split. This is a realistic and important lesson: **a single feature can still separate noisy data perfectly if the noise doesn't happen to cross that feature's chosen threshold.** Let's check a competing candidate to show how the algorithm actually compares options (it doesn't know in advance that CreditScore is "the right" feature):

**Candidate split: Income ≤ 62.5** (midpoint between 60 and 65)

- Left (Income ≤ 62.5): IDs {1,2,6,9} → all Yes → $Gini_L=0$
- Right (Income > 62.5): IDs {3,4,5,7,8,10} → {No,**Yes**,No,No,No,No} = 1 Yes, 5 No → $p_{Yes}=1/6$

$$
Gini_R = 1-\left(\left(\tfrac16\right)^2+\left(\tfrac56\right)^2\right) = 1-(0.028+0.694)=0.278
$$
$$
Gini_{weighted} = \tfrac{4}{10}(0)+\tfrac{6}{10}(0.278) = 0.167
$$
$$
\Delta Gini = 0.5-0.167 = 0.333
$$

**Step 5 in action**: comparing $\Delta Gini = 0.5$ (CreditScore split) vs. $0.333$ (Income split) — the algorithm selects **CreditScore ≤ 667.5** as the root split, since it strictly dominates. This is exactly how step 4→5 plays out across *every* feature at *every* node — in a real implementation, dozens of candidate thresholds across all features are scored this way and the single best is kept.

Since both children of the CreditScore split are already pure ($Gini=0$), step 2's stopping criterion ("is node pure?") halts recursion immediately — this is a **1-level, 2-leaf tree** that already achieves 100% training accuracy on this toy set. This is precisely the kind of result that should make you suspicious in real practice: a tree this shallow fitting training data perfectly on noisy real-world data is a strong overfitting signal, not a strong model signal (more in 1.6).

## 1.5 Worked example — Regression split (House Price)

**Root node** (all 10 samples), $\bar y = \tfrac{180+250+320+150+270+400+165+310+120+350}{10} = 251.5$

$$
\text{Var}_{root} = \tfrac{1}{10}\sum (y_i-251.5)^2 = \tfrac{1}{10}(5112.25+2.25+4692.25+10302.25+342.25+22052.25+7482.25+3422.25+17292.25+9702.25) = \tfrac{1}{10}(80402.5) = 8040.25
$$

**Candidate split: SqFt ≤ 16.5** (midpoint between 15 and 18)

- Left (SqFt ≤ 16.5): IDs {1,2,4,5,7,9} → prices {180,250,150,270,165,120}, $\bar y_L = 189.17$
$$
\text{Var}_L = \tfrac16\left[(180-189.17)^2+(250-189.17)^2+(150-189.17)^2+(270-189.17)^2+(165-189.17)^2+(120-189.17)^2\right]
$$
$$
= \tfrac16(84+3701+1535+6534+584+4784) = \tfrac16(17222)=2870.3
$$

- Right (SqFt > 16.5): IDs {3,6,8,10} → prices {320,400,310,350}, $\bar y_R = 345$
$$
\text{Var}_R = \tfrac14\left[(320-345)^2+(400-345)^2+(310-345)^2+(350-345)^2\right] = \tfrac14(625+3025+1225+25)=1225
$$

$$
\text{Var}_{weighted} = \tfrac{6}{10}(2870.3)+\tfrac{4}{10}(1225) = 1722.2+490=2212.2
$$
$$
\Delta\text{Var} = 8040.25-2212.2 = 5828.05
$$

This is the regression-tree analogue of step 4–5: every candidate threshold across SqFt, Age, and Bedrooms is scored this way (weighted variance after the split), and SqFt ≤ 16.5 would be compared against, say, Age- or Bedroom-based splits the same way Income lost to CreditScore above. The leaf prediction once recursion stops is simply $\bar y$ of whatever samples land there — this is why a regression tree's predictions look like a **staircase function**: piecewise-constant, with jump discontinuities at every learned threshold, never a smooth surface. That staircase nature is itself a limitation we'll return to.

## 1.6 Generalization, overfitting, and pruning

A fully grown tree (recursing until every leaf is pure or has 1 sample) achieves **zero training error** but typically **terrible test error** — this is the textbook low-bias/high-variance regime. Two control mechanisms exist:

**Pre-pruning (early stopping during growth)** — stop step 2 early via:
- `max_depth`: hard cap on tree depth
- `min_samples_split` / `min_samples_leaf`: refuse splits that would create tiny leaves
- `min_impurity_decrease`: refuse splits below a $\Delta$impurity threshold

**Post-pruning (cost-complexity / "weakest link" pruning)** — grow the full tree, then prune back using a regularized objective:

$$
R_\alpha(T) = \sum_{m=1}^{|T|}\sum_{i\in R_m} L(y_i,c_m) + \alpha|T|
$$

where $|T|$ is the number of leaves and $\alpha \geq 0$ controls the complexity penalty. For each candidate subtree obtained by collapsing internal nodes, compute this penalized cost; increasing $\alpha$ from 0 produces a *sequence* of nested subtrees, and cross-validation picks the $\alpha$ (and hence subtree) with the best validation performance.

**This $\alpha|T|$ penalty term is worth remembering precisely** — when we reach XGBoost, its objective function will contain a structurally identical term ($\gamma T$), except XGBoost bakes the penalty *into the greedy growth criterion itself* rather than applying it after the fact. That's a direct, traceable line of algorithmic evolution.

## 1.7 Missing values and surrogate splits

CART's specific answer to missing data — **surrogate splits** — works as follows:

1. At each node, after choosing the primary best split (e.g., CreditScore ≤ 667.5), CART also identifies one or more **surrogate splits**: other features whose own best threshold *produces the most similar partition* of the samples that *do* have both features observed (measured by agreement in how samples are routed left/right).
2. At prediction or training time, if a sample is missing the primary split's feature, the **best surrogate that the sample does have data for** is used to decide its routing instead.
3. If a sample is missing *every* surrogate too, it falls back to the **majority direction** (whichever child got more training samples).

Why this matters: surrogate splits let the tree handle missing data **without imputation** and without discarding samples, by exploiting natural correlations between features (e.g., if Income is missing, DebtRatio might be highly correlated with the same underlying creditworthiness signal and routes the sample almost as well). The tradeoff: surrogate splits add meaningful training cost (you're solving a small split-finding problem for every candidate surrogate feature, at every node, not just for the primary feature), which is one reason later libraries (XGBoost, LightGBM) chose a cheaper alternative — learning a **default direction** for missing values directly as part of split-gain optimization (covered in their respective sections) rather than CART's surrogate-correlation approach.

## 1.8 Training complexity

For $n$ samples and $d$ features:
- Pre-sorting each feature once costs $O(n\log n)$ per feature → $O(d\, n\log n)$ total, reusable across the whole tree if sorted indices are maintained cleverly (this is what efficient implementations like sklearn's do).
- At each node, evaluating all candidate thresholds for all features costs $O(d \cdot n_{node})$ (a single linear scan over sorted values, accumulating running sums of class counts/target sums).
- Summed across all nodes at a given depth, the work is $O(d\cdot n)$ per depth level (since nodes at one depth partition the full dataset).
- For a balanced tree of depth $D = O(\log n)$: total cost $\approx O(d\, n\log n)$.
- For a **degenerate/imbalanced tree** (depth approaching $n$, which is possible if min-samples constraints are loose): worst case degrades to $O(d\, n^2)$.

This is the practical reason `max_depth` and `min_samples_leaf` aren't purely about overfitting — they're also a **direct lever on training time**, which becomes critical once you're growing thousands of trees in a boosting ensemble (motivating histogram-binning approaches in LightGBM, which converts this entirely from a sort-based to a bucket-counting problem).

## 1.9 Hyperparameters (the practical knobs)

| Hyperparameter | What it controls | Effect of increasing it |
|---|---|---|
| `max_depth` | Maximum tree depth | Tighter constraint: underfitting risk ↑, variance ↓ |
| `min_samples_split` | Min samples required to consider splitting a node | Tighter pruning, fewer tiny splits |
| `min_samples_leaf` | Min samples required in *each* resulting leaf | Smooths leaf predictions, prevents single-outlier leaves |
| `max_features` | Number of features considered at each split | Lower = more randomness, sets up Random Forest's mechanism directly |
| `min_impurity_decrease` | Minimum $\Delta$impurity required to accept a split | Filters out marginal/noise-driven splits |
| `ccp_alpha` | Cost-complexity pruning strength ($\alpha$ above) | Larger = more aggressive post-pruning, smaller final tree |
| `criterion` | Gini vs. entropy (classification), squared vs. absolute error (regression) | Usually minor effect (per Document 1, §9) |

## 1.10 Sample weighting

Every impurity calculation generalizes trivially to weighted samples — instead of raw counts, use **weighted sums**:

$$
p_i = \frac{\sum_{j: y_j=i} w_j}{\sum_j w_j}, \qquad \bar y_w = \frac{\sum_j w_j y_j}{\sum_j w_j}
$$

and Gini/entropy/variance formulas are applied identically on top of these weighted quantities. This is how `class_weight='balanced'` (common for imbalanced classification, e.g. fraud or rare-disease detection) and explicit `sample_weight` arrays work — a sample with weight 3 is treated, for splitting purposes, as if it were 3 identical copies, **without actually duplicating data and inflating memory/compute**. This same weighting mechanism resurfaces in a much more central role in boosting algorithms (AdaBoost *is*, essentially, a sample-reweighting algorithm at its core).

## 1.11 The weakness that motivates the next model

A single decision tree is a **high-variance, low-bias** estimator: it can represent very complex decision boundaries (low bias), but it is *unstable* — change a handful of training points (like our noisy row 4) and a different feature might win at the root, cascading into a structurally different tree with different predictions on unseen data. In the bias-variance decomposition, decision trees sit at the low-bias/high-variance end; pruning trades some variance for bias but never escapes the fundamentally unstable, greedy, single-shot nature of the estimation procedure.

**Random Forest's entire premise is a direct answer to this**: if you can generate many *decorrelated* high-variance trees and average their predictions, variance drops (averaging $k$ i.i.d.-ish estimators divides variance roughly by $k$) while bias stays low — without touching the tree-growing algorithm itself at all. That's Section 2.

---

# Section 2 — Random Forest

## 2.1 What it fixes and how

A single decision tree is unstable because every tree sees the exact same dataset and greedily commits to one structure. Two sources of instability:

1. **One dataset** — the root split is decided by a razor-thin margin sometimes, and a handful of different training points could flip it entirely, cascading a completely different tree structure.
2. **All features at every node** — every tree uses the same dominant feature at the root, so all trees end up highly correlated. Averaging correlated trees doesn't reduce variance much.

Random Forest attacks both with two injections of randomness that don't touch the tree-growing algorithm at all:

| Injection | Mechanism | What it breaks |
|---|---|---|
| **Bootstrap sampling** | Each tree trains on a different random draw-with-replacement of the training data (~63% unique samples) | Different trees see different data → different splits at the root and throughout |
| **Feature subsampling** | At each split node, only a random subset of features ($\sqrt{d}$ for classification, $d/3$ for regression) is considered | Trees can no longer all agree on the same dominant feature → structural diversity |

The result: an ensemble of trees that are each individually high-variance but are *decorrelated* from each other. Averaging decorrelated high-variance estimators is the core mathematical mechanism behind why RF works.

## 2.2 The variance math — why decorrelation is the lever

For a single tree, prediction variance is $\sigma^2$. For the average of $B$ trees:

$$
\text{Var}\!\left(\frac{1}{B}\sum_{b=1}^B f_b(x)\right) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2
$$

where $\rho$ = average pairwise correlation between trees. Read this equation carefully:

- **First term $\rho\sigma^2$**: the irreducible floor — even with $B \to \infty$ trees, this remains. It's controlled *only* by how correlated the trees are with each other.
- **Second term $\frac{1-\rho}{B}\sigma^2$**: the reducible part — it shrinks as you add more trees, but only in proportion to $1-\rho$.

Two extreme cases:
- $\rho = 1$ (all trees identical, e.g. trained on the same data with no feature subsampling): variance = $\sigma^2$ — you added trees but got zero variance reduction.
- $\rho = 0$ (completely independent trees): variance = $\sigma^2/B$ — halves with every doubling of tree count.

**The practical consequence**: feature subsampling targets the $\rho\sigma^2$ term by forcing structural diversity. Bootstrap sampling primarily helps the $\frac{1-\rho}{B}\sigma^2$ term by generating different training sets. Both are needed. If you use bootstrap but no feature subsampling (that's just "Bagging"), trees share the same dominant feature at every root and $\rho$ stays high — you get much less improvement than full RF.

**Bias** is not improved by this process. Averaging trees doesn't reduce individual tree bias. The mean of $B$ trees with the same systematic error still has that systematic error. This is the fundamental ceiling of RF — and the gap that boosting later fills.

## 2.3 The algorithm — 8-step template

| Step | What happens | Notes |
|---|---|---|
| **1. Set ensemble size** | Choose $B$ = `n_estimators` | More trees = lower variance; returns diminish after ~200–500 |
| **2. For each tree $b = 1 \ldots B$:** | — | Steps 3–7 repeat inside this loop |
| **3. Bootstrap sample** | Draw $n$ samples with replacement from the training set → $D_b$ | Each $D_b$ contains ~63.2% unique training samples; the remaining ~36.8% are **out-of-bag (OOB)** |
| **4. Grow a full tree on $D_b$** | Apply CART recursively, but with step 5 added at every node | Standard CART except for the feature subsampling injection |
| **5. Feature subsampling at every split** | At each node, randomly draw $m$ features from all $d$ features; only these $m$ are candidates for that node's split | $m = \sqrt{d}$ (classification), $m = d/3$ (regression) by default; this is the decorrelation mechanism |
| **6. Find the best split** | Among the $m$ candidate features, pick the split maximizing Gini/variance reduction as in CART §1.3 steps 3–5 | Identical to CART, just on a restricted feature set |
| **7. Grow to stopping criteria** | Recurse until `max_depth`, `min_samples_leaf`, or pure nodes | RF typically uses very deep trees (low bias per tree); bias is not the concern, variance is |
| **8. Aggregate predictions** | **Classification**: majority vote across all $B$ trees. **Regression**: mean of all $B$ tree outputs | The averaging/voting step is where variance reduction actually occurs |

## 2.4 Worked example — Classification (Loan Default)

**Full training set** (10 samples): IDs 1–10, 5 Default=Yes, 5 Default=No.

### Bootstrap sampling — what each tree actually sees

Each of 3 illustrative trees draws 10 samples with replacement:

| Tree | IDs drawn | Unique IDs | OOB (unseen) | Noisy ID 4 status |
|---|---|---|---|---|
| **Tree 1** | {1,2,3,3,5,6,7,8,9,10} | {1,2,3,5,6,7,8,9,10} | **{4}** | Excluded — tree never sees the borderline point |
| **Tree 2** | {1,2,4,4,5,6,7,8,9,10} | {1,2,4,5,6,7,8,9,10} | **{3}** | Included twice — upweighted in this tree's training |
| **Tree 3** | {1,2,3,4,5,6,6,8,9,10} | {1,2,3,4,5,6,8,9,10} | **{7}** | Included once — seen normally |

This is the key mechanism: the noisy borderline sample (ID 4, CreditScore=645, Default=Yes) is excluded from roughly 37% of trees, included normally in another ~37%, and duplicated/down-weighted across the rest. The ensemble averages over all these scenarios rather than committing to one tree that always sees ID 4 exactly once.

### Feature subsampling — structural diversity across trees

With $d=3$ features and $m = \sqrt{3} \approx 2$, at each node 2 of {Income, CreditScore, DebtRatio} are drawn:

| Tree | Root split candidates drawn | Root split chosen | Structure |
|---|---|---|---|
| **Tree 1** (no ID 4) | {CreditScore, DebtRatio} | CreditScore ≤ 667.5 | Perfect split — both leaves pure |
| **Tree 2** (ID 4 ×2) | {Income, DebtRatio} | DebtRatio ≤ 0.275 | CreditScore not available → must use correlated proxy |
| **Tree 3** (ID 4 ×1) | {CreditScore, Income} | CreditScore ≤ 667.5 | Similar to Tree 1, but different bootstrap data |

Tree 2 *cannot* use CreditScore at the root (it wasn't drawn in the random subset). It must split on DebtRatio or Income instead. This forces structural diversity — Tree 2 captures a different facet of the data (debt load as a signal) rather than just being a near-duplicate of Tree 1.

### Aggregating votes on a new loan application

New sample: Income = 62k, CreditScore = 662, DebtRatio = 0.33 — deliberately borderline.

| Tree | Routing logic | Prediction |
|---|---|---|
| Tree 1 | CreditScore 662 ≤ 667.5 → left (all-Yes leaf) | **Yes** |
| Tree 2 | DebtRatio 0.33 > 0.275 → right (mostly-Yes leaf) | **Yes** |
| Tree 3 | CreditScore 662 ≤ 667.5 → left (all-Yes leaf) | **Yes** |

Majority vote: 3/3 → **Default = Yes** (confident). Now compare this to the same prediction from a single tree that happened to split on Income instead — it might route CreditScore=662 to a No leaf if Income=62 was the root feature. The RF is more robust because it's averaging over multiple views of the same borderline applicant.

For **probability estimates** (useful for threshold tuning with imbalanced classes): instead of majority vote, take the fraction of trees that predicted Yes — here 3/3 = 1.0, though in real datasets with noisy bootstrap samples this would be something like 0.72, which is actionable as a calibrated risk score.

## 2.5 Worked example — Regression (House Price)

**Goal**: predict price for a new house (SqFt=16, Age=7, Bedrooms=3).

Each tree is trained on a different bootstrap sample and — because it sees slightly different data and different features at each split — makes a different prediction:

| Tree | Bootstrap emphasized | Root split used | Prediction for new house |
|---|---|---|---|
| Tree 1 | Large houses upweighted | SqFt ≤ 16.5 → left child, $\bar y_L = 189k$ | **$205k**|
| Tree 2 | Older houses upweighted | Age ≤ 7.5 → left child, $\bar y_L = 298k$ | **$285k** |
| Tree 3 | Balanced sample | SqFt ≤ 16.5 → left child, but different leaf mean | **$220k** |
| Tree 4 | New houses upweighted | Bedrooms ≤ 3.5 → left child | **$240k** |
| Tree 5 | Mixed | SqFt ≤ 13.5 → right, Age next split | **$260k** |

**RF prediction** = mean across all trees = $(205 + 285 + 220 + 240 + 260) / 5 = \mathbf{\$242k}$

The individual tree predictions range from 205k to 285k — high variance per tree. The average is stable in the middle. Each tree is wrong in a different direction (one over-weighted large houses, one over-weighted old houses), and those errors tend to cancel out in the average. This cancellation of uncorrelated errors is the regression analogue of majority vote in classification.

## 2.6 Out-of-bag (OOB) evaluation

**The math**: when drawing $n$ samples with replacement, the probability any specific sample is never drawn is $(1-\frac{1}{n})^n \to e^{-1} \approx 0.368$ as $n \to \infty$. So each tree leaves roughly **36.8% of training samples unused** — these are the out-of-bag samples for that tree.

**How OOB scoring works**:
1. For each training sample $i$, collect only the predictions from trees that did *not* include sample $i$ in their bootstrap (i.e., sample $i$ was OOB for those trees).
2. Aggregate those predictions (majority vote / mean) to get $\hat y_i^{OOB}$.
3. Compare $\hat y_i^{OOB}$ against the true $y_i$ across all training samples → this is the OOB score.

**Why this matters**: OOB score is a free, nearly-unbiased estimate of generalization error — you get a validation metric without holding out a separate validation set. It's structurally similar to leave-one-out cross-validation but much cheaper to compute. In practice, OOB accuracy and 5-fold CV accuracy are very close for RF.

**Practical use**: set `oob_score=True` in sklearn. Monitor it alongside training error — a large gap signals overfitting; a poor OOB score despite hyperparameter tuning suggests the problem requires more data or feature engineering, not more trees.

## 2.7 Feature importance

Two fundamentally different approaches, with very different biases:

### MDI — Mean Decrease in Impurity (default in sklearn)

For each feature $f$, sum the weighted Gini/variance reductions it produced across all split nodes across all trees, normalized by the number of trees:

$$
\text{MDI}(f) = \frac{1}{B}\sum_{b=1}^{B} \sum_{\substack{v \in \text{Tree}_b \\ \text{split on } f}} \frac{n_v}{n} \cdot \Delta\text{Impurity}(v)
$$

**Advantages**: computed for free during training, no extra inference passes needed, fast.

**Known bias**: MDI systematically inflates the importance of high-cardinality features (continuous features or categoricals with many unique values). The reason is the same as ID3's high-cardinality bias from §6 of the basics doc — a feature with many possible thresholds has more chances to find a "lucky" impurity reduction purely by chance, and MDI accumulates all of those. **In the loan default example**: if you added a "CustomerID" column (unique per row), MDI might rank it highly because it can perfectly split individual samples — yet it's completely useless for prediction.

**When to trust MDI**: when features have similar cardinality (e.g., all are continuous with similar ranges), or when you're using it for *relative* ranking within a model rather than absolute importance.

### Permutation Importance (MDA — Mean Decrease in Accuracy)

For each feature $f$:
1. Compute baseline score (accuracy/MSE) on the OOB set.
2. Randomly shuffle feature $f$'s values in the OOB set, breaking its relationship with the target.
3. Re-score the model on the shuffled data. Record the drop in score.
4. Repeat several times and average.

$$
\text{Permutation Importance}(f) = \overline{\text{Score}_{\text{baseline}} - \text{Score}_{\text{shuffled}_f}}
$$

**Advantages**: unbiased with respect to cardinality, works with any evaluation metric (not tied to Gini), reveals whether a feature is *actually useful for prediction* vs. just statistically associated with impurity reduction.

**Disadvantages**: $O(d \times n_{\text{trees}})$ extra inference passes — expensive for large $d$. Also, if two features are highly correlated, shuffling one doesn't break its information (the correlated partner still carries it), so importance gets split between them and both look less important than they actually are.

**Rule of thumb for interviews**: MDI is fast and good for quick exploration. Permutation importance is better for feature selection decisions and communicating feature relevance to stakeholders. For production feature selection, use permutation importance.

### SHAP values (brief mention)

SHAP (SHapley Additive exPlanations) is a more theoretically grounded approach rooted in game theory — it assigns each feature a contribution that satisfies consistency, local accuracy, and missingness axioms that MDI and permutation importance do not. `shap` library computes TreeSHAP efficiently for RF and all tree ensembles. This is the gold standard for explainability in production tree model deployments, and worth being familiar with even though the full derivation belongs in an explainability-focused section.

## 2.8 Hyperparameters

| Hyperparameter | What it controls | Default | Practical guidance |
|---|---|---|---|
| `n_estimators` | Number of trees in the ensemble | 100 | Increase until OOB error stabilizes. Diminishing returns past ~300–500. More trees = slower predict-time, not just train-time. |
| `max_features` | Features sampled per split | `'sqrt'` (clf), `1.0` (reg) | Most important RF-specific parameter. Lower → more decorrelation, lower variance but higher bias per tree. `'sqrt'` / `'log2'` are standard. Try tuning this first. |
| `max_depth` | Max depth of each tree | `None` (full) | RF works best with deep trees (low bias). Only restrict to reduce memory or training time. |
| `min_samples_split` | Min samples to attempt a split | 2 | Increase (e.g. 5–20) to smooth leaf regions and reduce overfit on noisy data. |
| `min_samples_leaf` | Min samples required in each leaf | 1 | Increase for regression tasks (e.g. 3–10) to stabilize leaf mean estimates. More impactful than `min_samples_split`. |
| `bootstrap` | Whether to use bootstrap sampling | `True` | `False` = "pasting" (sampling without replacement). Disabling bootstrap removes OOB capability and reduces diversity. Almost always keep `True`. |
| `oob_score` | Compute OOB generalization score | `False` | Set `True` to get a free validation metric. Near-equivalent to CV in practice. |
| `class_weight` | Sample weights per class | `None` | `'balanced'`: $w_c = n/(k \cdot n_c)$. `'balanced_subsample'`: recomputes weights fresh per bootstrap sample (better for RF than plain `'balanced'`). Use for imbalanced labels. |
| `max_samples` | Samples per bootstrap draw | `None` (= $n$) | Set < $n$ to reduce training time at cost of diversity. Rarely needed unless $n$ is very large. |
| `max_leaf_nodes` | Max leaves per tree | `None` | Alternative to `max_depth` for controlling tree size. Limits total nodes, not depth specifically. |
| `min_impurity_decrease` | Min impurity reduction to accept a split | 0.0 | Useful for pruning splits that contribute noise. Start small (e.g. 1e-4) and increase. |
| `n_jobs` | Parallel workers | 1 | Set `-1` to use all cores. Trees are **embarrassingly parallel** — each is independent — so RF scales linearly with cores. One of its major advantages over sequential boosting. |
| `random_state` | Seed for reproducibility | `None` | Always set in production/experiments for reproducibility. |

**Tuning priority order**: `max_features` → `n_estimators` → `min_samples_leaf` → `max_depth` → `class_weight`. Most RF performance gain comes from the first two.

## 2.9 Missing values

**Sklearn RF does not handle missing values natively** — it raises an error if NaN appears in training or prediction data. This is a direct gap compared to CART's surrogate splits (§1.7).

Options in practice, in increasing order of sophistication:

**Simple imputation** (fastest): fill NaN with mean/median/mode before passing to RF. Easy to implement, but loses the "missingness as a signal" pattern (sometimes the fact that a field is missing is itself informative, e.g. a borrower who didn't report income).

**Missingness indicator** (better): add a binary column `feature_was_missing` alongside the imputed column. The RF can learn to split on the indicator first, effectively learning when missingness matters.

**MissForest** (best, expensive): an iterative algorithm that uses the RF itself to impute missing values — initialize with median imputation, train RF on complete rows, predict missing values, update, repeat until convergence. Handles non-linear correlations between features during imputation, unlike simple approaches.

**Contrast with later algorithms**: XGBoost and LightGBM both handle missing values natively by learning an optimal *default direction* for each split (left or right) from the training data — faster and more integrated than surrogate splits. CatBoost also handles missing values natively. This is one practical reason production systems often prefer XGBoost/LightGBM over RF when data quality is imperfect.

## 2.10 Imbalanced classes

**The problem**: if Default=No is 90% of data and Default=Yes is 10%, a tree that always predicts No gets 90% accuracy — but that's useless. The split criterion (Gini/entropy) naturally tends to favor splits that cleanly handle the majority class.

**Solution 1 — `class_weight='balanced_subsample'`** (recommended for RF specifically):

Computes class weights fresh inside each bootstrap sample:
$$
w_c = \frac{n_b}{k \cdot n_{b,c}}
$$
where $n_b$ = bootstrap sample size, $n_{b,c}$ = samples of class $c$ in that bootstrap. This adjusts the effective impurity calculation so minority class errors are penalized more heavily, without changing which samples appear — making it compatible with OOB scoring.

**Solution 2 — `class_weight='balanced'`**: uses global class weights computed once from the full training set. Simpler but doesn't account for class frequency variation across bootstrap samples.

**Solution 3 — Resampling before training**: oversample the minority class (SMOTE: generate synthetic minority samples by interpolating between existing ones) or undersample the majority class. Works independently of the model.

**Solution 4 — Threshold adjustment** (most flexible): train the RF normally, use `predict_proba()` to get probability scores, then adjust the classification threshold away from 0.5. Lowering the threshold for the minority class (e.g., predict Default=Yes if $P(\text{Yes}) > 0.3$) is equivalent to using class weights but gives you a continuous dial to tune. Evaluate with **PR-AUC** (precision-recall curve) rather than ROC-AUC for severely imbalanced problems — ROC-AUC is insensitive to class imbalance, PR-AUC is not.

## 2.11 Monotonic constraints

**RF does not support monotonic constraints natively** in scikit-learn. The ensemble average of many trees can *approximate* monotonicity on average, but individual trees can violate it locally, and there is no enforcement mechanism.

If monotonic constraints are required (e.g., "predicted default probability must be non-decreasing as DebtRatio increases, all else equal" — a regulatory requirement common in credit scoring), you need:
- **XGBoost**: `monotone_constraints` parameter — enforces during split selection.
- **LightGBM**: `monotone_constraints` parameter — same.
- **CatBoost**: `monotone_constraints` — same.

This is a meaningful practical limitation of RF in regulated industries. It's worth stating explicitly in an interview because it shows you've thought about deployment constraints, not just model accuracy.

## 2.12 Training complexity

| Component | Cost |
|---|---|
| Per tree (bootstrap + sort + grow) | $O(d_{\text{sub}} \cdot n \log n)$ where $d_{\text{sub}} = \sqrt{d}$ |
| Full ensemble of $B$ trees | $O(B \cdot \sqrt{d} \cdot n \log n)$ |
| Parallelism | Perfect — trees are fully independent; scales linearly with CPU cores |
| Memory | $O(B \cdot 2^{D_{\max}})$ — each tree stored independently; can dominate for large $B$, deep $D$ |
| Prediction per sample | $O(B \cdot D_{\max})$ — traverse one path through each tree |

The embarrassingly parallel training is RF's biggest systems-level advantage over boosting. With `n_jobs=-1`, RF training scales almost linearly with CPU count. This makes RF significantly faster in wall-clock time on multi-core machines even if boosting's sequential pass is computationally similar per-tree. At inference time, both are fast.

## 2.13 The weakness that motivates the next model

RF is excellent at variance reduction but has a **bias ceiling it cannot break**. Every tree in the ensemble uses the same greedy CART algorithm with the same structural limitation — axis-aligned, piecewise-constant predictions. Averaging many such trees still produces axis-aligned, piecewise-constant predictions with the same systematic errors. If the true decision boundary requires, say, capturing an interaction that no individual tree can represent efficiently, no amount of averaging will fix it.

More precisely: RF's bias equals the bias of a single tree (since $\mathbb{E}[\bar f] = \mathbb{E}[f_b]$ — the expected value of the average equals the expected value of any one tree). Pruning a tree increases bias; keeping trees deep keeps bias low but variance high; RF handles the variance side but leaves bias exactly where a single deep tree leaves it.

**Boosting algorithms attack the bias directly**: instead of training trees in parallel on different random subsets and averaging, they train trees *sequentially*, where each new tree specifically corrects the errors of the ensemble so far. The first tree makes predictions; the second tree learns from the residuals of the first; the third from the residuals of the first two; and so on. Bias decreases with each additional tree — the opposite of RF's dynamic, which is variance decreasing with each additional tree.

**AdaBoost is the first and simplest instantiation of this idea**: rather than fitting trees to residuals directly, it reweights the training samples so that misclassified samples get higher weight — forcing the next tree to focus its effort on the examples the current ensemble gets wrong. That's Section 3.

---

# Section 3 — AdaBoost (Adaptive Boosting)

## 3.1 What it fixes and the core mechanism

RF reduces variance by training trees in parallel on random data subsets and averaging. But:
- **Bias stays constant** — the ensemble bias equals individual tree bias.
- **All trees equally weighted** — a weak tree's bad predictions have the same influence as a strong tree's good ones.

AdaBoost fixes both by **sequential reweighting**:

1. Train the first weak learner (shallow tree) on the original data.
2. Look at where it was wrong — misclassified samples.
3. **Reweight the training data** so misclassified samples get higher weight, correct samples get lower weight.
4. Train the second weak learner on this reweighted data. It naturally focuses on the examples the first one struggled with.
5. Repeat: each new tree targets the residual errors of the ensemble so far.
6. At prediction time, **weighted majority vote** — trees that were accurate on the training data get higher weight in the final prediction than trees that were less accurate.

**The critical difference from RF**: each new tree is *not* independent — it's explicitly correcting the mistakes of all previous trees. This is why bias decreases. The first tree captures the dominant signal; the second tree captures the next-strongest signal; and so on. Averaging many such trees, each specializing on different error modes, reduces bias (the systematic gap between the ensemble and the true function).

## 3.2 The objective function — classification case (AdaBoost.M1 for binary)

AdaBoost minimizes an **exponential loss** function:

$$
L_{\exp}(m) = \sum_{i=1}^n e^{-y_i \cdot F_m(x_i)}
$$

where:
- $F_m(x_i) = \sum_{b=1}^m \alpha_b h_b(x_i)$ is the cumulative ensemble prediction after $m$ trees.
- $y_i \in \{-1, +1\}$ (note: **not** {0, 1} like in standard sklearn — the conversion is internal).
- $h_b(x_i) \in \{-1, +1\}$ is the prediction of tree $b$ (class -1 or +1).
- $\alpha_b$ is the **weight (learning rate) assigned to tree $b$** — strong trees get higher $\alpha$.

**Why exponential loss?**

$$
e^{-y_i F_m} = \begin{cases}
e^{-F_m} & \text{if } y_i F_m > 0 \text{ (correct prediction)} \\
e^{+F_m} & \text{if } y_i F_m < 0 \text{ (incorrect prediction)}
\end{cases}
$$

Correct predictions on the right side of the margin (far from decision boundary) have exponentially low loss; incorrect predictions incur exponentially growing loss. **This margin-based penalty** is the mathematical core of AdaBoost: it fiercely penalizes confidently incorrect predictions, moderately penalizes uncertain ones, and ignores confidently correct ones.

## 3.3 The algorithm — 9-step template (classification, AdaBoost.M1)

| Step | What happens | Formula / Details |
|---|---|---|
| **1. Initialize** | Set all sample weights to uniform | $w_i^{(1)} = 1/n$ for all $i$ |
| **2. For tree $b = 1 \ldots B$:** | — | Outer loop over $B$ boosting iterations |
| **3. Train weak learner** | Fit a shallow tree (default: depth 1, a "stump") on weighted data | Tree grows on the data $(x_i, y_i)$ with sample weights $w_i^{(b)}$. Each weighted sample is treated as if it appears $w_i^{(b)}$ times, exactly like CART's sample weighting (§1.10) |
| **4. Compute weighted error** | Fraction of weighted samples that tree $b$ misclassified | $\epsilon_b = \sum_{i: h_b(x_i) \neq y_i} w_i^{(b)}$ |
| **5. Check stopping criterion** | If $\epsilon_b \geq 0.5$ or $\epsilon_b \leq 0$, discard tree and stop boosting | A tree worse than random (>0.5) is useless; a perfect tree (0) means the remaining errors are separated and further boosting is futile (or overfit) |
| **6. Compute tree weight** | How much to trust this tree's predictions in the final vote | $\alpha_b = \tfrac{1}{2}\ln\left(\frac{1-\epsilon_b}{\epsilon_b}\right)$ |
| **7. Update sample weights** | Reweight samples for the next tree to focus on errors | $w_i^{(b+1)} = w_i^{(b)} \cdot e^{-\alpha_b \cdot y_i \cdot h_b(x_i)} / Z_b$ where $Z_b = \sum_i w_i^{(b)} \cdot e^{-\alpha_b \cdot y_i \cdot h_b(x_i)}$ is a normalization constant |
| **8. Aggregate** | Combine all $B$ trees with their learned weights | — |
| **9. Final prediction** | Weighted majority vote using $\alpha$ weights | $F(x) = \sum_{b=1}^B \alpha_b h_b(x)$; predict Yes if $F(x) > 0$, No if $F(x) < 0$ |

**The reweighting formula in detail (step 7)**:

$$
w_i^{(b+1)} = w_i^{(b)} \cdot e^{-\alpha_b y_i h_b(x_i)}
$$

The exponent $y_i h_b(x_i)$ is:
- $+1$ (correct prediction) → exponent is $-\alpha_b$ → weight decreases (sample $i$ is downweighted)
- $-1$ (incorrect prediction) → exponent is $+\alpha_b$ → weight increases (sample $i$ is upweighted)

The magnitude of the change is $\alpha_b$ — strong trees ($\alpha_b$ large) cause bigger reweighting; weak trees cause smaller shifts. Then divide by $Z_b$ to keep weights normalized (sum to 1).

## 3.4 Worked example — Classification (Loan Default)

**Dataset**: 10 samples (Table from §1), training to depth-1 trees (stumps). Initialize $w_i^{(1)} = 0.1$ for all $i$.

---

### Iteration 1: First tree

**Weighted Gini scores** for all candidate stumps on the reweighted data. Since weights are uniform, this is the same as §1.4:

- CreditScore ≤ 667.5: Gini gain = 0.5 (perfect split)
- Income ≤ 62.5: Gini gain = 0.333
- DebtRatio ≤ 0.275: (some gain)

**Selected split**: CreditScore ≤ 667.5.

**Tree 1 predictions** (a stump with one split):
- CreditScore ≤ 667.5 → left leaf → predict "Yes" (all 5 left samples are Yes)
- CreditScore > 667.5 → right leaf → predict "No" (all 5 right samples are No)

**Misclassifications on training data**: 0 (perfect split on this noisy toy set). So $\epsilon_1 = 0$.

**Problem**: a perfect tree breaks the algorithm because $\alpha_1 = \tfrac{1}{2}\ln\left(\frac{1-0}{0}\right) = \infty$. This is actually overfit — the toy dataset is too small and pure. In practice, you'd either:
- Add `max_depth` constraint and retrain on realistic data where no single stump is perfect.
- Or build a slightly deeper tree.

Let's artificially assume this first stump misclassifies 1 sample (it makes a mistake on one borderline point): $\epsilon_1 = 0.1$.

$$
\alpha_1 = \tfrac{1}{2}\ln\left(\frac{1-0.1}{0.1}\right) = \tfrac{1}{2}\ln(9) = \tfrac{1}{2}\times 2.197 = 1.099
$$

**Reweight samples** using step 7:

Let's say the misclassified sample is ID 4 (borderline Default=Yes that the stump incorrectly predicted No):
- For ID 4 (misclassified): $w_4^{(2)} = 0.1 \cdot e^{1.099} / Z_1 = 0.1 \cdot 3.0 / Z_1 = 0.3/Z_1$ (upweighted by factor of 3)
- For all 9 correctly classified: $w_i^{(2)} = 0.1 \cdot e^{-1.099}/Z_1 = 0.1 \cdot 0.333/Z_1 = 0.0333/Z_1$ (downweighted by factor of 3)

Normalization: $Z_1 = 9 \times 0.1 \times 0.333 + 1 \times 0.1 \times 3.0 = 0.3 + 0.3 = 0.6$

After normalization:
- $w_4^{(2)} = 0.3 / 0.6 = 0.5$ (50% of total weight)
- $w_i^{(2)} = 0.0333 / 0.6 = 0.0556$ for the 9 others (5.56% each)

The second tree will see ID 4 as 50× heavier than all others — it's screaming "focus on this sample!"

---

### Iteration 2: Second tree

With the reweighted data, CreditScore ≤ 667.5 is now a much worse split — it puts ID 4 (now 50% of weight) into the "Yes" leaf along with other Yes samples, but it's a borderline case that should maybe be No. A different feature, Income or DebtRatio, might better separate the remaining error.

**Selected split** (hypothetically): DebtRatio ≤ 0.325

This stump makes a different decision boundary, catching errors the first stump missed. Suppose it misclassifies 2 of the reweighted samples (IDs 7, 8) with combined reweighted weight $\epsilon_2 = 0.15$.

$$
\alpha_2 = \tfrac{1}{2}\ln\left(\frac{0.85}{0.15}\right) = \tfrac{1}{2}\ln(5.67) = 0.844
$$

Slightly lower than $\alpha_1$, because this tree is weaker (15% error vs. 10%).

---

### Final prediction on a new loan (Income=62, CreditScore=662, DebtRatio=0.33):

| Tree | Stump criterion | Prediction | Weight $\alpha$ |
|---|---|---|---|
| Tree 1 | CreditScore ≤ 667.5 → True | Predict **Yes** | 1.099 |
| Tree 2 | DebtRatio ≤ 0.325 → False | Predict **No** | 0.844 |

**Weighted vote**: $F = 1.099 \times 1 + 0.844 \times (-1) = 0.255 > 0$ → **Predict Yes, but with less confidence than Tree 1 alone would give**.

The first tree said Yes; the second said No. The first tree's vote carries more weight (1.099 > 0.844) because it was more accurate on training data. The final ensemble is conservative about Yes — yes, but with doubt.

If you had three trees and one more voted Yes and two voted No, the weighted vote would average them. This weighted aggregation is the core of how AdaBoost transitions from "many weak learners" to a strong final model.

## 3.5 Algorithm template — Regression (AdaBoost.R2)

**AdaBoost.R2** (Drucker's formulation, 1997) extends boosting to regression by replacing margin-based loss with an error-based criterion and using a different reweighting scheme. The core idea is identical to classification — sequential boosting targeting residuals — but loss and updates differ.

| Step | What happens | Formula / Details |
|---|---|---|
| **1. Initialize** | Set all sample weights to uniform | $w_i^{(1)} = 1/n$ for all $i$ |
| **2. For tree $b = 1 \ldots B$:** | — | Outer loop over $B$ boosting rounds |
| **3. Train weak learner** | Fit a shallow regression tree on weighted data | Tree predicts continuous values; weighted samples are treated as before |
| **4. Compute predictions** | Get $\hat y_i^{(b)} = h_b(x_i)$ for all training samples | — |
| **5. Compute normalized errors** | Express each error relative to the maximum error in this round | $L_i^{(b)} = \dfrac{\|y_i - \hat y_i^{(b)}\|}{D_\infty^{(b)}}$ where $D_\infty^{(b)} = \max_i \|y_i - \hat y_i^{(b)}\|$ |
| **6. Compute weighted error** | Median of normalized errors (50th percentile) | $\epsilon_b = \text{median}_i L_i^{(b)}$ over all $i$ |
| **7. Check stopping criterion** | If $\epsilon_b = 0$ (perfect tree) or $\epsilon_b \geq 0.5$ (useless tree), stop | A tree worse than predicting the median is useless; perfection suggests overfitting |
| **8. Compute tree weight** | Weight this tree's predictions based on accuracy | $\alpha_b = \dfrac{\epsilon_b}{1-\epsilon_b}$; take $\beta_b = \ln(1/\alpha_b)$ (inverted: stronger trees have *lower* weight in the update rule, counterintuitive but correct) |
| **9. Update sample weights** | Reweight for the next tree | $w_i^{(b+1)} = w_i^{(b)} \cdot \alpha_b^{1-L_i^{(b)}}$ (normalized); samples with low error stay light, high error get heavier |
| **10. Aggregate** | Combine all predictions, weighted by tree strength | $F(x) = \dfrac{\sum_{b=1}^B \beta_b h_b(x)}{\sum_{b=1}^B \beta_b}$ (weighted mean) |

**Key differences from classification**:
- **Median error**, not 0-1 loss. Why median? It's robust — a single outlier error doesn't dominate the calculation (unlike mean error).
- **Inverted weight formula**: $\alpha_b = \epsilon_b/(1-\epsilon_b)$, so $\alpha_b < 1$ for good trees, $\alpha_b > 1$ for bad ones. Then $\beta_b = \ln(1/\alpha_b)$ flips it: strong trees get high $\beta$ weight. This inverse relationship is confusing but standard in Drucker's formulation.
- **Normalized error** based on max error, not margin: $L_i = |y_i - \hat y|/D_\infty$ scales errors to [0,1], making the algorithm adaptive to scale changes.
- **Weighted mean aggregation** at prediction time (stronger trees have more influence in the final prediction). This mirrors the classification case where stronger trees get higher $\alpha$ weights in the final vote.

## 3.5a Worked example — Regression (AdaBoost.R2, house price)

**Dataset**: 10 houses (Table from §2.5), training to depth-1 trees (stumps on single features).

---

### Iteration 1: First tree

**Equal weights**: $w_i^{(1)} = 0.1$ for all $i$.

**Best stump** (from §1.5): SqFt ≤ 16.5
- Left (SqFt ≤ 16.5): IDs {1,2,4,5,7,9}, predicted mean $\hat y_L = 189.17$k
- Right (SqFt > 16.5): IDs {3,6,8,10}, predicted mean $\hat y_R = 345.0$k

**Predictions and errors**:

| House | Actual | Predicted | Error $\|y-\hat y\|$ | Normalized $L_i = \|y-\hat y\|/D_\infty$ |
|---|---|---|---|---|
| 1 | 180 | 189.17 | 9.17 | 9.17/80.83 = 0.1134 |
| 2 | 250 | 189.17 | 60.83 | 60.83/80.83 = 0.7523 |
| 3 | 320 | 345 | 25 | 25/80.83 = 0.3091 |
| 4 | 150 | 189.17 | 39.17 | 39.17/80.83 = 0.4846 |
| 5 | 270 | 189.17 | 80.83 | 80.83/80.83 = 1.0000 |
| 6 | 400 | 345 | 55 | 55/80.83 = 0.6802 |
| 7 | 165 | 189.17 | 24.17 | 24.17/80.83 = 0.2990 |
| 8 | 310 | 345 | 35 | 35/80.83 = 0.4331 |
| 9 | 120 | 189.17 | 69.17 | 69.17/80.83 = 0.8555 |
| 10 | 350 | 345 | 5 | 5/80.83 = 0.0618 |

$D_\infty^{(1)} = \max\{9.17, 60.83, 25, 39.17, 80.83, 55, 24.17, 35, 69.17, 5\} = 80.83$ (House 5 prediction is worst).

Sorted normalized errors: {0.0618, 0.1134, 0.2990, 0.3091, 0.4331, 0.4846, 0.6802, 0.7523, 0.8555, 1.0000}.

**Weighted error** (median of the 10 normalized errors):
$$
\epsilon_1 = \text{median}(0.0618, 0.1134, ..., 1.0000) = \frac{0.4331 + 0.4846}{2} = 0.4588
$$

**Tree weight**:
$$
\alpha_1 = \frac{\epsilon_1}{1-\epsilon_1} = \frac{0.4588}{0.5412} = 0.848
$$
$$
\beta_1 = \ln(1/\alpha_1) = \ln(1.1792) = 0.1654
$$

So this first tree's contribution is multiplied by $\beta_1 = 0.1654$ in the final ensemble — a fairly weak tree, since median error is moderately high.

**Reweight samples** (step 9) using the exponential form $w_i^{(b+1)} = w_i^{(b)} \cdot e^{\beta_b L_i^{(b)}}$ with normalization:

Houses with high $L_i$ (large errors) get upweighted; those with low error stay light:
- House 5 ($L_5 = 1.0$, highest error — the max): $w_5^{(2)} \propto 0.1 \cdot e^{0.1654 \cdot 1.0} = 0.1 \cdot e^{0.1654} = 0.1 \cdot 1.1797 = 0.1180$ (upweighted by ~1.18×)
- House 10 ($L_{10} = 0.0618$, lowest error): $w_{10}^{(2)} \propto 0.1 \cdot e^{0.1654 \cdot 0.0618} = 0.1 \cdot e^{0.01022} = 0.1 \cdot 1.01027 = 0.1010$ (barely changed)
- House 2 ($L_2 = 0.7523$, second-worst): $w_2^{(2)} \propto 0.1 \cdot e^{0.1654 \cdot 0.7523} = 0.1 \cdot e^{0.1244} = 0.1 \cdot 1.1324 = 0.1132$ (upweighted by ~1.13×)

After normalization to sum to 1: the most error-prone houses (5, 2, 9) become heavier, pulling the next tree's attention toward them.

---

### Iteration 2: Second tree

With updated weights favoring Houses 5, 2, 9 (and others with high errors from Iteration 1), the next tree looks for a split that better predicts these high-error houses.

Perhaps a different feature like Age ≤ 7.5 (from §1.5 example) does better on the reweighted data. Suppose Tree 2 achieves:

$$
\epsilon_2 = 0.320 \quad(\text{better than Tree 1's } 0.4588)
$$
$$
\beta_2 = \ln(1/\alpha_2) = \ln\left(\frac{1}{0.463}\right) = \ln(2.16) = 0.768
$$

Tree 2 is moderately stronger ($\epsilon_2 < \epsilon_1$) and gets a higher weight ($\beta_2 = 0.768$ vs. $\beta_1 = 0.1654$).

---

### Prediction on a new house (SqFt=16, Age=7, Bedrooms=3)

| Tree | Feature split | Predicted price | Weight $\beta$ |
|---|---|---|---|
| Tree 1 | SqFt ≤ 16.5 → left | 189.17k | 0.1654 |
| Tree 2 | Age ≤ 7.5 → left | 298k | 0.768 |

**Final prediction** (weighted mean, normalized by $\beta$ values):
$$
F(x) = \frac{\sum_b \beta_b h_b(x)}{\sum_b \beta_b} = \frac{0.1654 \times 189.17 + 0.768 \times 298}{0.1654 + 0.768} = \frac{31.3 + 228.9}{0.9334} = \frac{260.2}{0.9334} = 278.6k
$$

Tree 2's prediction (298k) carries ~82% of the final weight ($\beta_2 / (\beta_1 + \beta_2) = 0.768/0.9334 = 0.823$) since it was the stronger tree. Tree 1's prediction (189.17k) carries only ~18%, reflecting its mediocre accuracy on the training data. If we had a third tree that was even stronger, it would dominate the weighted average.

---

### Why this is different from classification

In classification, the margin is a signed quantity, so early trees' errors become the target for later trees in a principled directional way. In regression, errors are unsigned, so AdaBoost.R2 uses a simpler strategy: "which samples were worst?" → reweight them → next tree focuses there. The median-error criterion and exponential reweighting reflect that regression doesn't have the margin structure.

## 3.6 The margin and classification confidence

One intuition for why exponential loss works: the quantity $y_i F_m(x_i)$ is called the **margin** — the signed prediction times the true label.

$$
\text{Margin}_i = y_i \cdot F_m(x_i) = y_i \sum_{b=1}^m \alpha_b h_b(x_i)
$$

- **Margin > 0** (correct prediction): the ensemble and truth agree on the sign.
- **Margin > 1** (correct, confident): the weighted vote is so skewed toward the correct class that even if you reduced the tree weights, you'd still be correct.
- **Margin close to 0** (uncertain): ensemble is nearly tied.
- **Margin < 0** (incorrect, confidently wrong): the ensemble is wrong.
- **Margin far negative** (incorrect, very confidently wrong): exponential loss rises sharply.

The exponential loss $e^{-\text{Margin}}$ is thus:
- Very small when margin is large (correct + confident → low loss).
- Moderate when margin is small (correct but uncertain → medium loss).
- Very large when margin is negative (incorrect → high loss, with exponential explosion for very negative margins).

This is the **margin-maximization principle** — AdaBoost implicitly tries to push predictions far from the decision boundary (margin → $\infty$), not just barely correct. This is theoretically elegant and leads to good generalization, though the exponential penalty can be brittle with outliers (a few very hard-to-classify examples can blow up the loss for later trees).

## 3.7 Hyperparameters

| Hyperparameter | What it controls | Default | Guidance |
|---|---|---|---|
| `n_estimators` | Number of boosting rounds (trees) | 50 | Start with 100–200; increase until validation error stops improving. Unlike RF, more trees can overfit if learning_rate is high. Monitor OOB/validation error carefully. |
| `learning_rate` (or `learning_rate`) | Shrinkage — multiplies each $\alpha_b$ by this factor | 1.0 | **Most important hyperparameter**. Reduce to 0.01–0.1 to slow down boosting, reduce variance, improve generalization. $\alpha_b \to \eta \cdot \alpha_b$. Typical trade-off: lower learning rate requires more trees to achieve same accuracy. |
| `base_estimator` | The weak learner type and depth | Decision tree stump (depth=1) | For classification: stump often optimal due to bias-variance trade-off (low-bias boosting + low-capacity base learner). For regression or harder problems, try depth 3–5. Deeper base learners increase the per-tree bias, reducing the potential for boosting to improve. |
| `loss` | Loss function for regression (AdaBoost.R2) | 'linear' | 'linear', 'square', 'exponential'. Linear is most robust; exponential most aggressive. Start with linear for regression. |
| `random_state` | Seed | None | Always set for reproducibility. |

**Learning rate vs. n_estimators trade-off**:
- High learning rate (1.0) + few trees (50): fast, but high variance, risk of overfitting.
- Low learning rate (0.01) + many trees (1000): slow training, but better generalization.
- In practice, sklearn's default (1.0) often overfits. A typical starting point is `learning_rate=0.1, n_estimators=200`.

## 3.8 Comparing weak learners

The effectiveness of boosting depends heavily on the **base learner capacity**:

| Base Learner | Depth | Typical error rate | When to use |
|---|---|---|---|
| Stump | 1 | 40–60% (just better than random) | Standard; low variance, interpretable boosting |
| Shallow tree | 3–5 | 20–40% | Problems where stumps plateau in accuracy; e.g., non-linear interactions |
| Linear model | — | 30–50% | Linear separability but with noise; very fast |
| Deep tree | 8+ | 1–5% (nearly memorizing) | Rarely — tends to overfit; boosting of deep trees acts like single tree ensemble, not correction |

The key principle: **AdaBoost works best with weak learners** — models with error rate just slightly better than random, around 40–60%. If the base learner is too strong (e.g., a full-depth tree), it's already capturing most of the signal, and boosting adds little value. If it's too weak (e.g., a linear model on highly nonlinear data), it can't improve much no matter how much boosting you do.

## 3.9 Missing values and sample weighting

**Missing values**: AdaBoost relies on sample weights, not explicit missing-value handling. So it inherits CART's gap — no native support for NaN.

Use the same strategies as RF (§2.9): simple imputation, missingness indicators, or MissForest.

**Sample weights**: AdaBoost computes sample weights internally as part of its algorithm (step 7). However, sklearn also allows user-provided `sample_weight` in the fit call, which are *initial* weights for the very first tree. The algorithm then applies exponential reweighting on top of those initial weights. This is useful for imbalanced classes:

```python
weights = np.where(y == 1, 1.0, 5.0)  # minority class weight 5x
ada.fit(X, y, sample_weight=weights)
```

**Imbalanced classes**: If you set initial weights proportional to class rarity, the first tree focuses on the minority class, and subsequent trees inherit those priorities through reweighting. This is more seamless than RF's `class_weight` parameter.

## 3.10 Training complexity

| Component | Cost |
|---|---|
| Per tree ($b$-th iteration) | $O(d \cdot n \log n)$ for sorting + $O(d \cdot n)$ for growing on reweighted data |
| Full ensemble of $B$ trees | $O(B \cdot d \cdot n \log n)$ (same per-tree cost, but sequential, not parallel) |
| Parallelism | **None** — each tree depends on the reweighting from the previous tree; strictly sequential |
| Prediction per sample | $O(B \cdot D_{\max})$ — traverse one path per tree |

**Sequential vs. parallel**: This is AdaBoost's main computational limitation. RF trains all $B$ trees independently (embarrassingly parallel), so wall-clock training time is ~$O(d \cdot n \log n)$ with $B$ cores. AdaBoost is strictly sequential — $B$ trees must be trained one after the other. So wall-clock time is roughly $O(B \cdot d \cdot n \log n)$ on a single core, and **gains almost nothing from multiple cores** (only per-tree parallelism is possible, e.g., within the tree-growing loop, which is not common in sklearn).

This is a major practical reason gradient boosting and its descendants (XGBoost, LightGBM) became more popular in production — they also perform sequential boosting, but their better regularization and systems-level parallelization make them vastly faster on large datasets.

## 3.11 Feature importance

Unlike RF's MDI (which sums Gini reductions), AdaBoost feature importance is computed as:

$$
\text{Importance}(f) = \frac{1}{B}\sum_{b=1}^{B} \alpha_b \cdot \text{Frequency}_b(f)
$$

where $\text{Frequency}_b(f)$ is the number of times feature $f$ appears in tree $b$ (or a binary indicator if it appears). The weight $\alpha_b$ accounts for tree strength — strong trees' feature usage is weighted more heavily in the final importance.

**Advantages**: naturally accounts for tree quality (weak trees are downweighted), less biased toward high-cardinality features than MDI (since frequency is discrete, not a continuous gain sum).

**Disadvantages**: still biased by feature interactions and multicollinearity (two correlated features split the importance). Permutation importance (same as RF's version) works on AdaBoost too and is generally preferable for production decisions.

## 3.12 The weakness that motivates the next model

AdaBoost's exponential loss, while elegant, has fundamental limitations:

1. **Outlier sensitivity**: a single misclassified outlier can cause $e^{-\text{Margin}}$ to explode, forcing the next tree to chase it at the expense of the bulk of the data. A robust method needs a loss that doesn't penalize outliers exponentially.

2. **Locked to exponential loss**: AdaBoost's theory and practice are tied to minimizing exponential loss for classification, and there's no canonical loss for regression. Extending to other loss functions (e.g., log-loss for probability calibration, or Huber loss for robustness) requires new variants (AdaBoost.L2, etc.), each with their own properties.

3. **Manual loss specification**: if you want to optimize AUC, log-loss, quantile loss, or any custom objective, AdaBoost gives you no framework — you have to engineer a new boosting algorithm.

**Gradient Boosting solves this** by inverting the problem: instead of committing to a specific loss and deriving sample reweighting from it, gradient boosting asks "what loss function do I want to minimize?" and then uses **gradient descent in function space** to construct the boosting procedure automatically. The result: a single, generalized framework that handles any differentiable loss, any type of target (classification, regression, ranking), and automatically derives the correct reweighting/correction strategy for each. That's Section 4.

---

# Section 4 — Gradient Boosting (GBM)

## 4.1 What it fixes and the core insight

AdaBoost is locked into exponential loss for classification and has no principled loss for regression. GBM inverts the architecture: instead of choosing a loss and deriving the boosting procedure, GBM asks **"what loss do I want to minimize?"** and then derives the boosting procedure automatically using **gradient descent in function space**.

The core idea: at iteration $m$, you have an ensemble $F_{m-1}(x)$ making predictions. Rather than reweighting samples (AdaBoost's approach), you:

1. **Compute the gradient of the loss** with respect to the current predictions: $g_i = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$. These are called **pseudo-residuals** — they point in the direction the predictions should move to reduce loss.
2. **Fit a regression tree** $h_m(x)$ to predict these pseudo-residuals (not the original targets).
3. **Update the ensemble**: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$, where $\eta \in (0,1)$ is the learning rate (shrinkage).

This is **gradient descent in function space**: instead of updating parameters, you're updating a function (the ensemble prediction) one gradient step at a time. The loss can be anything differentiable — squared error, log-loss, Huber loss, quantile loss — and the same algorithm works.

**Why this matters**:
- **Outlier robustness**: If you use Huber loss instead of squared error, outliers don't explode in importance — the algorithm adapts automatically.
- **Arbitrary objectives**: Want to optimize for AUC, logloss, or custom business metrics? Pick the loss, GBM handles the rest.
- **Unified framework**: Classification, regression, ranking, multi-task learning — all use the same algorithm.

## 4.2 The objective function

Formally, GBM minimizes:

$$
\hat F = \arg\min_F \sum_{i=1}^n L(y_i, F(x_i))
$$

where $L$ is any differentiable loss:
- **Regression (squared error)**: $L(y, \hat y) = \frac{1}{2}(y - \hat y)^2$
- **Binary classification (log-loss / cross-entropy)**: $L(y, p) = -[y \log p + (1-y) \log(1-p)]$ where $p = \text{sigmoid}(F(x))$
- **Huber loss (robust regression)**: $L(y, \hat y) = \begin{cases} \frac{1}{2}(y-\hat y)^2 & \text{if } |y-\hat y| \leq \delta \\ \delta(|y-\hat y| - \frac{\delta}{2}) & \text{otherwise} \end{cases}$

GBM solves this using **function-space gradient descent**. At each iteration:

$$
F_m(x) = F_{m-1}(x) + \eta \arg\min_h \sum_{i=1}^n L(y_i, F_{m-1}(x_i) + h(x_i))
$$

The inner optimization (finding $h$) is approximated by fitting a tree to the negative gradients (pseudo-residuals).

## 4.3 The algorithm — 10-step template (GBM, any loss)

| Step | What happens | Formula / Details |
|---|---|---|
| **1. Initialize** | Start with a constant prediction | $F_0(x) = \arg\min_c \sum_i L(y_i, c)$. For squared error: $F_0 = \bar y$. For log-loss: $F_0 = \log(\text{odds})$ (proportional to $\log(p/(1-p))$ where $p$ = proportion of positives) |
| **2. For iteration $m = 1 \ldots M$:** | — | Outer loop over $M$ boosting rounds |
| **3. Compute pseudo-residuals (negative gradients)** | For each training sample, compute the direction to reduce loss | $r_i^{(m)} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$ |
| **4. Fit a regression tree** | Fit a shallow tree (depth 3–5) to predict the pseudo-residuals | Tree $h_m(x)$ minimizes $\sum_i (r_i^{(m)} - h_m(x_i))^2$ (standard regression tree on gradient targets) |
| **5. Compute leaf outputs** | For each leaf, set the output value to optimize the loss in that region | $\gamma_{m,\ell} = \arg\min_\gamma \sum_{i \in \text{leaf}_\ell} L(y_i, F_{m-1}(x_i) + \gamma)$. For squared error: mean residual in leaf. For log-loss: Newton step (more complex) |
| **6. Apply shrinkage** | Scale the tree's contribution to prevent overfitting | $\tilde h_m(x) = \eta \cdot h_m(x)$ where $\eta \in (0,1)$ is the learning rate |
| **7. Update ensemble** | Add the scaled tree to the ensemble | $F_m(x) = F_{m-1}(x) + \tilde h_m(x)$ |
| **8. Monitor loss** | Optionally, track training and validation loss | Useful for early stopping if validation loss plateaus |
| **9. Aggregate** | After $M$ iterations, final ensemble is $F_M(x) = F_0(x) + \sum_{m=1}^M \eta h_m(x)$ | — |
| **10. Predict** | For regression: output $F_M(x)$ directly. For classification: output $p = \text{sigmoid}(F_M(x))$ or class $= \mathbf{1}[F_M(x) > 0]$ | — |

**Key insight (step 3–4)**: The pseudo-residual $r_i = -\partial L / \partial F$ is the direction in which the loss decreases fastest. Fitting a tree to these residuals is equivalent to taking a greedy gradient step. This is why GBM is "gradient descent in function space" — you're following the gradient, one tree at a time.

## 4.4 Worked example — Classification (Loan Default, log-loss)

**Dataset**: 10 samples (Table from §1), binary classification with log-loss.

Log-loss: $L(y, p) = -[y \log p + (1-y) \log(1-p)]$ where $p = \text{sigmoid}(F(x))$.

---

### Initialization

The optimal constant prediction $F_0$ minimizes the total log-loss. For binary classification with balanced classes, this is the log-odds:

$$
F_0 = \log\left(\frac{\text{# positives}}{\text{# negatives}}\right)
$$

In our dataset: 5 positives (Default=Yes) and 5 negatives (Default=No), so:

$$
F_0 = \log\left(\frac{5}{5}\right) = \log(1) = 0
$$

Thus $F_0(x) = 0$ for all samples, yielding $p_0 = \text{sigmoid}(0) = 0.5$ (neutral, 50% default probability for everyone).

---

### Iteration 1: First tree

**Compute pseudo-residuals** (negative gradients of the loss). 

For binary log-loss $L(y, F) = -[y \log p + (1-y) \log(1-p)]$ where $p = \text{sigmoid}(F)$, the gradient with respect to the ensemble's prediction $F$ is:

$$
\frac{\partial L}{\partial F} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial F} = (p - y)
$$

The **pseudo-residual** (the gradient direction pointing toward loss reduction) is the negative of this:

$$
r_i = -\frac{\partial L}{\partial F} = -(p_i - y_i) = y_i - p_i
$$

With $F_0 = 0$, all samples have $p_0 = 0.5$. So:

| Sample | ID | $y$ | $p_0$ | Gradient $(p_0-y)$ | Pseudo-residual $r_i = (y-p_0)$ |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 0.5 | $-0.5$ | $0.5$ |
| 2 | 2 | 1 | 0.5 | $-0.5$ | $0.5$ |
| 3 | 3 | 0 | 0.5 | $0.5$ | $-0.5$ |
| 4 | 4 | 1 | 0.5 | $-0.5$ | $0.5$ |
| 5 | 5 | 0 | 0.5 | $0.5$ | $-0.5$ |
| 6 | 6 | 1 | 0.5 | $-0.5$ | $0.5$ |
| 7 | 7 | 0 | 0.5 | $0.5$ | $-0.5$ |
| 8 | 8 | 0 | 0.5 | $0.5$ | $-0.5$ |
| 9 | 9 | 1 | 0.5 | $-0.5$ | $0.5$ |
| 10 | 10 | 0 | 0.5 | $0.5$ | $-0.5$ |

**Targets for the tree**: pseudo-residuals {0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5}.

**Fit a regression tree** to predict these residuals. This is a standard regression tree on the pseudo-residual targets (not the original $y$). Using the loan data features:

Best split (say): CreditScore ≤ 667.5
- Left (Default=Yes): IDs {1,2,4,6,9} → residuals {0.5, 0.5, 0.5, 0.5, 0.5} → mean = **0.5**
- Right (Default=No): IDs {3,5,7,8,10} → residuals {-0.5, -0.5, -0.5, -0.5, -0.5} → mean = **-0.5**

**Compute leaf outputs** (for log-loss, this is a Newton step, but for simplicity with symmetric data, the mean residual is optimal):
- Left leaf output: $\gamma_L = 0.5$
- Right leaf output: $\gamma_R = -0.5$

**Apply shrinkage** (learning rate $\eta = 0.1$):
- Scaled left output: $0.1 \times 0.5 = 0.05$
- Scaled right output: $0.1 \times (-0.5) = -0.05$

**Update ensemble**:
$$
F_1(x) = F_0(x) + \eta h_1(x) = 0 + 0.1 \times h_1(x)
$$

For a new sample routed to the left leaf: $F_1 = 0 + 0.05 = 0.05$ → $p_1 = \text{sigmoid}(0.05) \approx 0.512$ (shifted slightly toward Yes from 0.5).

---

### Iteration 2: Second tree

**Compute new pseudo-residuals** using $F_1$ instead of $F_0$. For sample 1 (Default=Yes, routed left):
$$
p_1 = \text{sigmoid}(0.05) = 0.5125
$$
$$
r_1^{(2)} = y_1 - p_1 = 1 - 0.5125 = 0.4875
$$

Similarly, all yes-samples now have residuals slightly below 0.5 (the first tree captured part of their signal, so the residual is smaller), and no-samples have residuals slightly above -0.5. The second tree targets this refined error pattern, making further incremental corrections.

**Fit a second tree** to the new residuals, find the best split (possibly a different feature or threshold), compute leaf outputs, apply shrinkage, and update the ensemble.

**Iteration 3 onwards**: Repeat, with each tree targeting the current residuals, gradually improving predictions.

---

### Final prediction on a new applicant (Income=62k, CreditScore=662, DebtRatio=0.33)

After $M = 100$ boosting rounds (typical):
$$
F_{100}(x) = 0 + 0.1 \times h_1(x) + 0.1 \times h_2(x) + \ldots + 0.1 \times h_{100}(x)
$$

If most trees route this borderline case to the Yes leaf (due to features), $F_{100} \approx 2.5$, giving:
$$
p_{100} = \text{sigmoid}(2.5) = \frac{1}{1+e^{-2.5}} \approx 0.924
$$

Predict **Default = Yes** with confidence ~92%.

---

### Why this works better than AdaBoost

1. **Gradient-based**: The algorithm directly follows the loss function's gradient. If log-loss is what you want, GBM optimizes it exactly.
2. **Outliers**: A single misclassified sample doesn't cause exponential explosion (like AdaBoost's $e^{-\text{Margin}}$). The gradient is bounded — even a very wrong prediction contributes a finite amount to the next tree's target.
3. **Flexibility**: Swap log-loss for Huber loss or custom loss, same algorithm. AdaBoost requires re-deriving everything.

## 4.5 Worked example — Regression (House Price, squared error)

**Dataset**: 10 houses (Table from §2.5), regression with squared error loss.

---

### Initialization

For squared error loss, the optimal constant prediction is simply the mean of all targets:

$$
F_0 = \arg\min_c \sum_{i=1}^n (y_i - c)^2 = \bar y
$$

Computing the mean of our house prices:

$$
F_0 = \frac{180+250+320+150+270+400+165+310+120+350}{10} = 251.5k
$$

All samples initially receive the constant prediction **$F_0(x) = 251.5k$** regardless of features.

---

### Iteration 1: First tree

**Compute pseudo-residuals** for squared error:

$$
\frac{\partial L}{\partial F} = -2(y - F) \quad \Rightarrow \quad r_i = -\frac{\partial L}{\partial F} = 2(y - F) \propto (y - \bar y)
$$

(The factor of 2 is usually dropped in practice; what matters is the residual direction.)

| House | Actual | $F_0$ | Residual $y - F_0$ |
|---|---|---|---|
| 1 | 180 | 251.5 | -71.5 |
| 2 | 250 | 251.5 | -1.5 |
| 3 | 320 | 251.5 | 68.5 |
| 4 | 150 | 251.5 | -101.5 |
| 5 | 270 | 251.5 | 18.5 |
| 6 | 400 | 251.5 | 148.5 |
| 7 | 165 | 251.5 | -86.5 |
| 8 | 310 | 251.5 | 58.5 |
| 9 | 120 | 251.5 | -131.5 |
| 10 | 350 | 251.5 | 98.5 |

**Fit a regression tree** to these residuals. Best split (say): SqFt ≤ 16.5
- Left (SqFt ≤ 16.5): IDs {1,2,4,5,7,9} → residuals {-71.5, -1.5, -101.5, 18.5, -86.5, -131.5} → mean = **-62.17**
- Right (SqFt > 16.5): IDs {3,6,8,10} → residuals {68.5, 148.5, 58.5, 98.5} → mean = **93.5**

**Compute leaf outputs** (for squared error, just the mean residual):
- Left leaf: $\gamma_L = -62.17$
- Right leaf: $\gamma_R = 93.5$

**Apply shrinkage** ($\eta = 0.1$):
- Scaled left: $0.1 \times (-62.17) = -6.217$
- Scaled right: $0.1 \times 93.5 = 9.35$

**Update ensemble**:
$$
F_1(x) = 251.5 + h_1(x)
$$

For a small house (left leaf): $F_1 = 251.5 - 6.217 = 245.28k$ (adjusted down from 251.5, pulling toward the true average of small houses ~189k, but conservatively by only 6.2k).

For a large house (right leaf): $F_1 = 251.5 + 9.35 = 260.85k$ (adjusted up toward large homes ~345k, but conservatively).

---

### Iteration 2: Second tree

**New residuals** using $F_1$ instead of $F_0$. Small houses now have residuals slightly smaller in magnitude (the prior step captured part of their deficit), and large houses similarly.

The second tree targets the remaining errors, making further refinements.

After enough iterations (e.g., $M = 100$), predictions are well-calibrated across the feature space.

---

### Why this is better than single trees or random forests

1. **Bias reduction**: Each tree explicitly corrects prior errors. Unlike RF (which trains independent trees), every new tree is a corrective step.
2. **Flexible loss**: If you want quantile regression (minimize absolute error), Huber loss (robust to outliers), or custom business loss, just change the loss function.
3. **Calibrated predictions**: Because you're following the true loss gradient, the ensemble is naturally well-calibrated to the objective you care about.

## 4.6 Learning rate and number of iterations

Two key hyperparameters control the bias-variance trade-off:

$$
F_M(x) = F_0(x) + \eta \sum_{m=1}^M h_m(x)
$$

**Learning rate $\eta$ (shrinkage)**:
- High $\eta$ (e.g., 0.5): large steps, fewer trees needed, higher variance.
- Low $\eta$ (e.g., 0.01): small steps, many trees needed, lower variance, better generalization (typically).

**Number of iterations $M$**:
- Few trees: underfitting, high bias.
- Many trees with low $\eta$: overfitting (if no regularization), but learning rate keeps variance controlled.

**Typical settings**: $\eta = 0.05$ to $0.2$, $M = 100$ to $1000$. The product $\eta \times M$ controls total "distance traveled" in function space — with low $\eta$, you need more trees to reach the same point.

**Early stopping**: Monitor validation loss and stop when it plateaus, rather than training a fixed $M$. This is one of GBM's big advantages — you don't need to commit to $M$ in advance.

## 4.7 Hyperparameters (GBM in scikit-learn and XGBoost)

| Hyperparameter | What it controls | Default (sklearn) | Guidance |
|---|---|---|---|
| `learning_rate` | Shrinkage factor $\eta$ | 0.1 | Lower (0.01–0.05) → more trees, better generalization. Higher (0.1–0.5) → faster training, often overfits. Start with 0.1. |
| `n_estimators` | Number of boosting rounds $M$ | 100 | Increase until validation error plateaus. Typical range 100–1000. Use early stopping to avoid guessing. |
| `max_depth` | Max depth of each tree | 3 | GBM works best with shallow trees (3–8). Deep trees (>10) often overfit. Shallow trees = weak learners that benefit from boosting. |
| `min_samples_leaf` | Min samples in each leaf | 1 | Increase (5–20) to smooth predictions, reduce overfit. For regression, especially important. |
| `subsample` | Fraction of samples per iteration | 1.0 | Set < 1.0 (e.g., 0.8) to inject randomness, reduce variance. Stochastic boosting. |
| `max_features` | Features considered per split | None (all) | Set < 1.0 (e.g., 0.8) to reduce correlation between trees. Stochastic GBM. |
| `loss` | Loss function | 'deviance' (log-loss for binary) | 'deviance' (classification), 'exponential' (rare), 'huber' (robust regression), etc. |
| `validation_fraction` | Fraction of data for validation (early stopping) | 0.1 | Hold out 10–20% for validation. Monitor validation loss; stop when it stops improving. |
| `n_iter_no_change` | Early stopping rounds | None | E.g., 10: if validation loss doesn't improve for 10 consecutive iterations, stop. Critical for preventing overfitting. |

**Tuning strategy**: 
1. Start with `learning_rate=0.1`, `n_estimators=200`, `max_depth=3`.
2. Use `n_iter_no_change` for early stopping.
3. If underfitting (high training and validation error): increase `max_depth` or `n_estimators`.
4. If overfitting (low training error, high validation error): decrease `learning_rate`, increase `subsample` / `max_features`.

## 4.8 Training complexity

| Component | Cost |
|---|---|
| Per iteration (tree $m$) | $O(d \cdot n \log n)$ for sorting features + $O(d \cdot n)$ for growing tree |
| Full ensemble ($M$ iterations) | $O(M \cdot d \cdot n \log n)$ sequential |
| Parallelism | Per-tree parallelism only; inherently sequential (unlike RF) |
| Prediction per sample | $O(M \cdot D_{\max})$ — traverse $M$ trees |

**Sequential bottleneck**: GBM is strictly sequential — tree $m$ depends on residuals from trees $1 \ldots m-1$. Training time is $O(M)$ times a single tree. With $M = 100$ to 1000 iterations, this can be slow on large datasets.

This is the key limitation GBM has vs. RF, and it's the primary motivation for **XGBoost and LightGBM** (Section 5–6), which introduce:
- **Histogram-based splits** (LightGBM): instead of sorted features, bucket them into 256 bins, reducing sorting cost.
- **Second-order (Newton) approximation** (XGBoost): uses Hessian to make fewer, more accurate splits.
- **Parallel learning** (LightGBM): multiple trees per iteration using leaf-wise growth.

## 4.9 Feature importance

GBM feature importance is computed the same way as AdaBoost — sum of weighted gains:

$$
\text{Importance}(f) = \frac{1}{M} \sum_{m=1}^{M} \sum_{\substack{v \in \text{Tree}_m \\ \text{split on } f}} w_v \cdot \Delta\text{Loss}(v)
$$

where $w_v$ is the weight of node $v$ (typically proportion of samples reaching that node) and $\Delta\text{Loss}$ is the loss reduction.

**Advantages over RF's MDI**: Each tree is solving a well-defined task (reduce residual variance in the data), so splits are more meaningful. Strong trees' features are weighted more heavily.

**Limitations**: Still biased by feature interactions and multicollinearity. Use permutation importance for production feature selection decisions.

## 4.10 Missing values and sample weighting

**Missing values**: GBM (in scikit-learn) does not handle NaN natively — use the same strategies as CART and AdaBoost (imputation, missingness indicators, MissForest).

However, **XGBoost and LightGBM handle missing values natively** by learning a default direction for each split (left or right) — a major practical advantage over vanilla GBM.

**Sample weighting**: GBM supports per-sample weights in the loss function:

$$
L = \sum_{i=1}^n w_i L(y_i, F(x_i))
$$

For imbalanced classification, set $w_i = $ weight inversely proportional to class frequency. This is handled in the gradient computation: $r_i = w_i \cdot (-\partial L / \partial F)$.

## 4.11 The weakness that motivates the next model

GBM is powerful and general, but has practical limitations:

1. **No built-in regularization**: The objective function is just $\sum L(y_i, F(x_i))$. There's no term penalizing tree complexity, so regularization happens only through learning rate and early stopping. With noisy data or limited validation set, overfitting is easy.

2. **Slow on large data**: Every iteration requires sorting all $d$ features across all $n$ samples for split search. With $n = 1M$ and $d = 1000$, sorting dominates training time.

3. **No native categorical handling**: Categorical features must be encoded (one-hot, label encoding), which is clunky and error-prone. Missing values aren't handled natively either.

4. **No explicit second-order optimization**: GBM uses the first-order gradient (negative residual) to pick splits. A second-order (Newton) approximation would make splits more targeted, especially early in boosting when residuals are large.

**XGBoost (Section 5) addresses points 1–4** by adding:
- Explicit L1/L2 regularization terms in the objective.
- Histogram-based splits and parallelization (faster).
- Native handling of missing values via default directions.
- Second-order Taylor approximation for split scoring.

That's the path forward.

---

# Section 5 — XGBoost (Extreme Gradient Boosting)

## 5.1 What it fixes and the core improvements

GBM is mathematically elegant but has practical limitations in three areas:

1. **No regularization** — overfits easily on noisy data; relies on early stopping as the only control.
2. **Slow on large data** — sorting is $O(n \log n)$ per feature per iteration; with $n = 10M$ samples, this is prohibitive.
3. **First-order approximation only** — the gradient (residual) guides splits, but ignores curvature (second derivatives), which could identify better splits faster.

**XGBoost** (Chen & Guestrin, 2016) fixes all three:

- **Regularized objective**: Adds $\gamma T + \frac{\lambda}{2}\sum_j w_j^2$ terms explicitly penalizing tree complexity and leaf weights.
- **Second-order Taylor approximation**: Uses both first and second derivatives ($g_i$ and $h_i$) in split scoring, enabling exact closed-form gain calculations.
- **Systems-level engineering**: Histogram-based split search (bucket features instead of sorting), parallel tree building, GPU support.

The result: XGBoost is typically 10–100× faster than GBM on large datasets while achieving equal or better accuracy due to better regularization.

## 5.2 The objective function

XGBoost minimizes:

$$
\mathcal{L}(F) = \sum_{i=1}^n L(y_i, F(x_i)) + \sum_{m=1}^M \Omega(h_m)
$$

where the regularization term is:

$$
\Omega(h_m) = \gamma T_m + \frac{\lambda}{2}\sum_{j=1}^{T_m} w_j^2
$$

**Breaking this down**:
- $T_m$ = number of leaves in tree $m$
- $\gamma$ = complexity penalty per leaf (higher → penalize tree complexity more)
- $w_j$ = output weight (prediction value) of leaf $j$
- $\lambda$ = L2 regularization on leaf weights (higher → shrink weights toward zero)

**Intuition**: 
- $\gamma T_m$ discourages creating many leaves (simpler trees win).
- $\frac{\lambda}{2}\sum w_j^2$ discourages large leaf weights (conservative predictions win).
- Together, they prevent overfitting while GBM has no such safeguards.

This is **structurally similar to CART's cost-complexity pruning** (§1.6, $\alpha|T|$), except XGBoost bakes it *into the split-scoring criterion itself* rather than applying it after tree growth. This makes regularization **integrated** rather than **post-hoc**.

## 5.3 Split gain with second-order approximation

At each node, XGBoost must score candidate splits using the **regularized gain formula**. This is where the second-order approximation enters.

Using a **Taylor expansion** of the loss around the current prediction $F_{m-1}$, the optimal leaf weight is:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

where:
- $G_j = \sum_{i \in \text{leaf}_j} g_i$ = sum of first derivatives (gradients)
- $H_j = \sum_{i \in \text{leaf}_j} h_i$ = sum of second derivatives (Hessian)
- The denominator $H_j + \lambda$ incorporates the L2 regularization

And the **regularized gain** (information gain accounting for regularization) is:

$$
\text{Gain}_{\text{reg}} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma
$$

**Compare to GBM's gain** (which is just variance reduction on pseudo-residuals $r_i = -g_i$, ignoring $h_i$):

$$
\text{Gain}_{\text{GBM}} = \text{Var}(r) - \left(\frac{|L|}{|D|}\text{Var}(r_L) + \frac{|R|}{|D|}\text{Var}(r_R)\right)
$$

**Why second-order matters**:
- When residuals are large (early boosting), the curvature $h_i$ is different across samples — high-curvature samples benefit from different treatment.
- The Hessian captures this curvature; ignoring it (as GBM does) leads to suboptimal splits early on.
- XGBoost's formula naturally weights high-confidence regions (high $H_j$, low uncertainty) differently from uncertain regions, leading to faster convergence.

## 5.4 The algorithm — 11-step template (XGBoost, any loss)

| Step | What happens | Details |
|---|---|---|
| **1. Initialize** | Constant prediction minimizing loss | Same as GBM: $F_0 = \arg\min_c \sum L(y_i, c)$ |
| **2. For iteration $m = 1 \ldots M$:** | — | Outer boosting loop |
| **3. Compute first derivatives** | Gradient of loss w.r.t. current prediction | $g_i = \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$ |
| **4. Compute second derivatives** | Hessian (second derivative) of loss | $h_i = \frac{\partial^2 L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)^2}$ |
| **5. Grow tree with regularized splitting** | For each node, find best split using regularized gain formula | Evaluate all candidate splits; pick $(feature^*, threshold^*) = \arg\max \text{Gain}_{\text{reg}}$ |
| **6. Compute optimal leaf weights** | For each leaf, set weight to minimize regularized loss in that region | $w_j = -\frac{G_j}{H_j + \lambda}$ |
| **7. Apply shrinkage** | Scale tree's contribution | $\tilde h_m(x) = \eta \cdot h_m(x)$ |
| **8. Update ensemble** | Add regularized tree to ensemble | $F_m(x) = F_{m-1}(x) + \tilde h_m(x)$ |
| **9. Prune leaves** | Remove leaves with negative contribution (post-growth pruning) | If $\text{Gain} < \gamma$ for a split, merge its children back |
| **10. Monitor loss** | Track training and validation loss, optionally stop early | — |
| **11. Predict** | Output final ensemble predictions | $F_M(x)$; for classification, apply sigmoid |

**Key differences from GBM** (steps 3–9):
- Steps 3–4 compute both $g_i$ and $h_i$; GBM only uses $g_i$.
- Step 5 uses the regularized gain formula instead of variance reduction.
- Step 6 explicitly optimizes leaf weights using the closed-form solution, not just the mean residual.
- Step 9 does post-growth pruning — removes splits that don't earn back their complexity penalty $\gamma$.

## 5.5 Worked example — Classification (Loan Default, log-loss with regularization)

**Dataset**: 10 samples (Table from §1), binary classification with log-loss.

**Hyperparameters**: $\lambda = 1.0$ (L2 weight regularization), $\gamma = 0.5$ (complexity penalty per leaf).

---

### Initialization

Same as GBM: $F_0 = 0$ (log-odds), all $p_0 = 0.5$.

---

### Iteration 1: First tree

**Compute gradients and Hessians**. For log-loss with $p = \text{sigmoid}(F)$:

$$
g_i = p_i - y_i, \quad h_i = p_i(1-p_i)
$$

With $p_0 = 0.5$ for all:

| Sample | ID | $y$ | $g_i$ | $h_i$ |
|---|---|---|---|---|
| 1 | 1 | 1 | -0.5 | 0.25 |
| 2 | 2 | 1 | -0.5 | 0.25 |
| 3 | 3 | 0 | 0.5 | 0.25 |
| ... | ... | ... | ... | ... |
| 10 | 10 | 0 | 0.5 | 0.25 |

All $h_i = 0.25$ (since $p=0.5 \implies p(1-p) = 0.25$), and gradients are ±0.5 as before.

---

**Candidate split: CreditScore ≤ 667.5**

Left leaf (5 Yes samples): $G_L = 5 \times (-0.5) = -2.5$, $H_L = 5 \times 0.25 = 1.25$

Right leaf (5 No samples): $G_R = 5 \times 0.5 = 2.5$, $H_R = 5 \times 0.25 = 1.25$

**Compute regularized gain**:

$$
\text{Gain}_{\text{reg}} = \frac{1}{2}\left[\frac{(-2.5)^2}{1.25+1} + \frac{(2.5)^2}{1.25+1} - \frac{0^2}{1.25+1.25+1}\right] - 0.5
$$

$$
= \frac{1}{2}\left[\frac{6.25}{2.25} + \frac{6.25}{2.25} - 0\right] - 0.5 = \frac{1}{2}[2.778 + 2.778] - 0.5 = 2.778 - 0.5 = 2.278
$$

(This is a **positive gain** after accounting for the complexity penalty, so the split is worth it.)

**Compute optimal leaf weights**:

$$
w_L = -\frac{G_L}{H_L + \lambda} = -\frac{-2.5}{1.25+1} = \frac{2.5}{2.25} = 1.111
$$

$$
w_R = -\frac{G_R}{H_R + \lambda} = -\frac{2.5}{1.25+1} = \frac{-2.5}{2.25} = -1.111
$$

**Apply shrinkage** ($\eta = 0.1$):

$$
\tilde w_L = 0.1 \times 1.111 = 0.1111, \quad \tilde w_R = 0.1 \times (-1.111) = -0.1111
$$

**Update ensemble**:

$$
F_1(x) = 0 + h_1(x) = \begin{cases} 0.1111 & \text{if CreditScore} \leq 667.5 \\ -0.1111 & \text{otherwise} \end{cases}
$$

For a Yes-sample routed left: $p_1 = \text{sigmoid}(0.1111) \approx 0.527$ (shifted toward Yes, more aggressively than GBM's 0.512 because XGBoost's regularized gain leads to larger leaf weights).

---

### Iteration 2: Second tree

**Compute new gradients and Hessians** using $F_1$ instead of $F_0$.

For a Yes-sample routed left: $p_1 \approx 0.527$

$$
g_i = p_1 - y = 0.527 - 1 = -0.473, \quad h_i = p_1(1-p_1) = 0.527 \times 0.473 \approx 0.249
$$

The gradients are now smaller in magnitude (first tree absorbed part of the signal), and the Hessians are slightly different from 0.25 (curvature changes as predictions move away from 0.5).

**Fit second tree** on the updated $(g_i, h_i)$ pairs, find the best regularized split, compute weights, apply shrinkage, update ensemble.

---

### Why XGBoost's second-order approach helps here

In iteration 1, GBM and XGBoost both achieved the same split (CreditScore), but:

- **GBM** set leaf outputs to the mean residual: ±0.5 (no curvature information).
- **XGBoost** set leaf outputs to the optimal weight incorporating curvature: ±1.111 (using Hessian).
- After shrinkage, XGBoost's step is larger: ±0.1111 vs GBM's ±0.05.

This means XGBoost converges faster — it takes fewer iterations to reach the same final prediction because each step is more informed by the loss surface's curvature.

---

## 5.6 Worked example — Regression (House Price, squared error with regularization)

**Dataset**: 10 houses (Table from §2.5).

**Hyperparameters**: $\lambda = 1.0$, $\gamma = 1.0$.

---

### Initialization

Same as GBM: $F_0 = \bar y = 251.5k$.

---

### Iteration 1: First tree

**Compute gradients and Hessians**. For squared error $L = (y - F)^2$:

$$
g_i = 2(F - y) = 2(251.5 - y_i), \quad h_i = 2 \text{ (constant for all samples)}
$$

| House | Actual | $g_i = 2(251.5-y)$ | $h_i$ |
|---|---|---|---|
| 1 | 180 | $2(251.5-180) = 143$ | 2 |
| 2 | 250 | $2(251.5-250) = 3$ | 2 |
| 3 | 320 | $2(251.5-320) = -137$ | 2 |
| 4 | 150 | $2(251.5-150) = 203$ | 2 |
| 5 | 270 | $2(251.5-270) = -37$ | 2 |
| 6 | 400 | $2(251.5-400) = -297$ | 2 |
| 7 | 165 | $2(251.5-165) = 173$ | 2 |
| 8 | 310 | $2(251.5-310) = -117$ | 2 |
| 9 | 120 | $2(251.5-120) = 263$ | 2 |
| 10 | 350 | $2(251.5-350) = -197$ | 2 |

(Note: the factor of 2 is often omitted in practice; it cancels in the gain formula anyway.)

---

**Candidate split: SqFt ≤ 16.5**

Left (small homes): IDs {1,2,4,5,7,9}

$$
G_L = 143 + 3 + 203 + (-37) + 173 + 263 = 748
$$
$$
H_L = 6 \times 2 = 12
$$

Right (large homes): IDs {3,6,8,10}

$$
G_R = -137 + (-297) + (-117) + (-197) = -748
$$
$$
H_R = 4 \times 2 = 8
$$

**Compute regularized gain** ($\lambda = 1.0, \gamma = 1.0$):

$$
\text{Gain}_{\text{reg}} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma
$$

$$
= \frac{1}{2}\left[\frac{748^2}{12+1} + \frac{(-748)^2}{8+1} - \frac{0^2}{12+8+1}\right] - 1.0
$$

$$
= \frac{1}{2}\left[\frac{559504}{13} + \frac{559504}{9}\right] - 1.0 = \frac{1}{2}[43039 + 62167] - 1.0 = 52603 - 1.0 \approx 52602
$$

(Enormous gain, as expected for a feature that cleanly separates house sizes.)

**Compute optimal leaf weights**:

$$
w_L = -\frac{G_L}{H_L+\lambda} = -\frac{748}{13} \approx -57.54
$$

$$
w_R = -\frac{G_R}{H_R+\lambda} = -\frac{-748}{9} \approx 83.11
$$

(These are adjustments to $F_0 = 251.5k$: small homes need to be pulled down by ~57.54, large homes pulled up by ~83.11.)

**Apply shrinkage** ($\eta = 0.1$):

$$
\tilde w_L = 0.1 \times (-57.54) = -5.754, \quad \tilde w_R = 0.1 \times 83.11 = 8.311
$$

**Update ensemble**:

$$
F_1(x) = 251.5 + h_1(x) = \begin{cases} 251.5 - 5.754 = 245.75k & \text{if SqFt} \leq 16.5 \\ 251.5 + 8.311 = 259.81k & \text{otherwise} \end{cases}
$$

For a small house (1800 sq ft, Actual=180k): $F_1 = 245.75k$ (pulled down from 251.5k toward the true small-home average of ~189k, but conservatively).

---

### Comparison to GBM

GBM would set leaf outputs to the mean residual:
- Left: mean of $(y - 251.5)$ = mean of {-71.5, -1.5, -101.5, 18.5, -86.5, -131.5} = -62.17
- Right: mean of {68.5, 148.5, 58.5, 98.5} = 93.5

XGBoost's regularized weights:
- Left: -57.54 (vs GBM's -62.17) — slightly less aggressive
- Right: 83.11 (vs GBM's 93.5) — slightly less aggressive

The difference is small here because the Hessian is constant (squared error has constant curvature), but in classification (where Hessian varies), XGBoost's second-order approach produces much sharper improvements.

---

## 5.7 Missing values and default direction

**The problem**: GBM doesn't natively handle NaN. XGBoost learns a **default direction** — when a sample is missing a feature, route it left or right based on which direction reduces loss more.

**How it works**:

1. During split evaluation, XGBoost considers three candidate splits:
   - **Route missing samples left** → compute gain assuming all NaN samples go left
   - **Route missing samples right** → compute gain assuming all NaN samples go right
   - **Skip this feature** → ignore it

2. Pick whichever default direction gives the highest gain.

3. At prediction time, if a sample has a missing value for this feature, route it to the learned default direction.

**Advantage**: No imputation needed, no information loss. Missing values become a learned routing decision, just like $\leq$ threshold for continuous features.

**Example**: In the loan dataset, if CreditScore has missing values:
- Try: route NaN left (toward Yes-heavy group) — compute gain
- Try: route NaN right (toward No-heavy group) — compute gain
- Pick the direction with higher gain
- At prediction, NaN values always route that way

This is elegant and data-efficient compared to GBM's imputation strategies.

## 5.8 Categorical features (one-hot encoding alternative)

**GBM limitation**: Categorical features must be one-hot encoded before fitting, which explodes dimensionality and loses ordinal information.

**XGBoost approach**: Natively handles categorical features by evaluating all possible partitions of categories in the split search (not just binary thresholds).

For a categorical feature with $k$ unique values, instead of:
- One-hot: $k$ binary columns
- Ordinal encoding: treating as continuous (loses category structure)

XGBoost considers splits like:
- **Category in {A, B, D}?** (any subset) vs. **Category in {C, E}?** (the rest)

This preserves the category structure and avoids explosion of columns.

**In practice** (XGBoost in sklearn): Set `max_cat_to_onehot` to control when to use native categorical vs. one-hot encoding. For production, native categorical is preferable.

## 5.9 Hyperparameters (XGBoost-specific and advanced)

This section covers the full hyperparameter space, organized by function. XGBoost has 50+ parameters; here are the critical ones for interviews.

### 5.9.1 Booster type — which base learner?

**`booster`** (default: `'gbtree'`) — determines the type of base learner:

| Booster | How it works | When to use | Tradeoff |
|---|---|---|---|
| **`'gbtree'`** | Gradient boosting with decision trees. Standard. Trains $M$ sequential trees, each correcting prior residuals. | 99% of use cases. Default. Handles nonlinear relationships, interactions. | Requires careful regularization. Can overfit on noisy data. |
| **`'gblinear'`** | Gradient boosting with linear base learners. Each iteration fits a linear regression on residuals. | When features are already preprocessed / linearly separable. When you want interpretability (coefficients). | Can't capture interactions or nonlinearities. Often worse accuracy than gbtree unless data is actually linear. |
| **`'dart'`** | **D**roput **A**dditive **R**egression **T**rees. Like gbtree but randomly "drops" prior trees when training new trees (similar to dropout in neural networks). Reduces overfitting. | When you have enough data and want robustness without early stopping. When boosting severely overfits. | Slower to train (dropout overhead). Results less stable (randomized). Harder to tune. |

**Example**: 
- **`booster='gbtree'`**: `xgb.XGBClassifier(booster='gbtree', max_depth=5)` — standard tree boosting.
- **`booster='gblinear'`**: If your loan default model only has Income and CreditScore (both numeric, limited interactions), linear boosting might work: `xgb.XGBClassifier(booster='gblinear')` learns coefficients like $0.05 \times \text{Income} - 0.01 \times \text{CreditScore}$ per iteration.
- **`booster='dart'`**: `xgb.XGBClassifier(booster='dart', rate_drop=0.1)` randomly drops 10% of prior trees when training each new tree — prevents early trees from dominating.

**Interview point**: "GBtree is default because trees are nonlinear feature learners. Gblinear is for when data is already feature-engineered. Dart is a regularization alternative to early stopping — instead of stopping training, it randomly forgets old trees."

---

### 5.9.2 Tree construction and splitting

**`tree_method`** (default: `'auto'`) — how XGBoost searches for splits:

| Method | Algorithm | Memory | Speed | Best for |
|---|---|---|---|---|
| **`'exact'`** | Full sort of each feature; evaluates all thresholds. XGBoost's reference implementation. | $O(n \log n)$ per iteration | Slow; ~seconds per tree on 1M rows | Small datasets (< 100K rows). Accuracy-critical, time-unconstrained. |
| **`'approx'`** | Quantile sketching. Buckets features into percentiles (~32 buckets), evaluates only bucket boundaries as thresholds. | $O(n)$ per iteration | Medium; ~10× faster than exact | Medium datasets (100K–10M rows). Balanced speed/accuracy. |
| **`'hist'`** | Histogram-based (same as LightGBM). Pre-buckets features into fixed bins (default 256) before training. Fastest. | $O(\text{bins} \times n)$ per iteration | Fast; 10–100× faster than exact | Large datasets (> 10M rows). GPU acceleration. |
| **`'gpu_hist'`** | Histogram-based on GPU. Builds histograms in parallel on GPU memory. | GPU VRAM-bound | Fastest (100–1000× for GPU-friendly operations) | Very large data with GPU available (RTX 3090, A100, etc.) |

**Sampling method** (interacts with `tree_method`):

**`sampling_method`** (default: `'uniform'`) — how samples are selected per iteration (for histogram methods):

| Method | How it works | Effect |
|---|---|---|
| **`'uniform'`** | Use all $n$ samples each iteration. | Stable, unbiased. Standard. |
| **`'gradient_based'`** | Like GOSS (LightGBM): keep samples with high-magnitude gradients, randomly sample low-gradient samples. | Faster (~30–50% data reduction per iteration). More variance, higher overfitting risk. |

**`max_bin`** (default: 256) — number of histogram bins per feature:

- **Higher max_bin** (512, 1024): finer granularity, more accurate splits (closer to exact), slower training.
- **Lower max_bin** (32, 64): coarser buckets, faster training, less accurate splits.
- **Practical**: 256 is almost always optimal. Only increase if data has many unique feature values and time/memory allow.

**Example**:
```python
# Large data (100M rows): use hist with 256 bins
xgb.XGBClassifier(tree_method='hist', max_bin=256, n_estimators=100)

# Very large data on GPU: use gpu_hist
xgb.XGBClassifier(tree_method='gpu_hist', max_bin=256, gpu_id=0)

# Small data (10K rows): use exact for best splits
xgb.XGBClassifier(tree_method='exact', n_estimators=100)
```

---

### 5.9.3 Tree structure regularization

| Hyperparameter | Controls | Default | Guidance |
|---|---|---|---|
| `max_depth` | Max tree depth | 6 | Start at 5–6. Increase if underfitting; decrease if overfitting. XGBoost handles deeper trees better than GBM due to regularization. Range: 3–15. |
| `min_child_weight` | Min sum of Hessian in a leaf (classification: proportional to sample count; regression: $\propto$ variance) | 1 | Raise to 3–5 for classification, 0.1–1 for regression. Prevents splitting on small, noisy subgroups. |
| `gamma` | Complexity penalty per split (from objective: $\gamma T$) | 0 | Increase (0.1–2) to reduce number of splits. If $\text{Gain} < \gamma$, split is rejected. Direct tree size control. |
| `lambda` | L2 weight regularization | 1.0 | Increase (1–10) to shrink leaf outputs toward zero. Prevents extreme predictions. Start at 1.0, tune if overfitting persists. |
| `alpha` | L1 weight regularization | 0 | Increase (0.1–10) to zero out weak leaf weights entirely. Less common than lambda; useful for feature selection. |

---

### 5.9.4 Stochasticity and variance reduction

| Hyperparameter | Controls | Default | Guidance |
|---|---|---|---|
| `subsample` | Fraction of samples used per tree | 1.0 | Set to 0.8 (80% of samples per iteration) for stochastic boosting. Reduces variance, slight bias increase. Speeds up training. |
| `colsample_bytree` | Fraction of features per tree | 1.0 | Set to 0.8–1.0. Reduces correlation between trees. Typical: 0.8. |
| `colsample_bylevel` | Fraction of features per split | 1.0 | Fine-grained control; rarely needed. Usually leave at 1.0. |
| `colsample_bynode` | Fraction of features per node (finer than bylevel) | 1.0 | Rarely tuned. Use if colsample_bylevel is not flexible enough. |

**Example**:
```python
# Stochastic boosting: use 80% of samples, 80% of features per tree
xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.8)

# More aggressive stochasticity (higher variance reduction, higher overfitting risk):
xgb.XGBClassifier(subsample=0.5, colsample_bytree=0.5)
```

---

### 5.9.5 Grow policy and tree structure

**`grow_policy`** (default: `'depthwise'`) — in histogram mode, how to expand the tree:

| Policy | Behavior | Memory | Convergence |
|---|---|---|---|
| **`'depthwise'`** (or `'lossguide'` for LightGBM-style) | Grows level-by-level. All nodes at depth $d$ before depth $d+1$. Balanced trees. | Higher (many intermediate nodes) | Slower (each iteration expands many nodes) |
| **`'lossguide'`** | Grows the single node with highest loss reduction (leaf-wise). Narrower, deeper trees. | Lower (fewer nodes) | Faster (fewer iterations needed) |

XGBoost's default `depthwise` is safe; `lossguide` is faster on large data but riskier for overfitting.

---

### 5.9.6 Objective functions and loss

**`objective`** (default: depends on problem) — the loss function to optimize:

**Classification**:

| Objective | Loss | When to use | Example |
|---|---|---|---|
| `'binary:logistic'` | Log-loss / cross-entropy | Binary classification, balanced classes | `objective='binary:logistic'` |
| `'binary:logitraw'` | Log-loss but outputs raw score (not probability) | Binary classification; skip sigmoid | Rarely used |
| `'multi:softmax'` | Softmax cross-entropy | Multiclass (> 2 classes) | `objective='multi:softmax', num_class=3` |
| `'multi:softprob'` | Softmax, outputs probability distribution per class | Multiclass with probability outputs | `objective='multi:softprob', num_class=3` |

**Regression**:

| Objective | Loss | When to use | Example |
|---|---|---|---|
| `'reg:squarederror'` | Squared error (MSE) | Standard regression, Gaussian errors | `objective='reg:squarederror'` |
| `'reg:pseudohuberloss'` | Pseudo-Huber loss: $\frac{(y - \hat y)^2}{\sqrt{1 + (y - \hat y)^2}}$ | Regression with some outliers. Smooth approximation to Huber. | `objective='reg:pseudohuberloss'` |
| `'reg:quantileerror'` | Quantile regression loss (asymmetric) | Predict percentiles, not means. E.g., 90th percentile price. | `objective='reg:quantileerror', quantile_alpha=0.9` |
| `'reg:absoluteerror'` | Mean absolute error (MAE) | Regression; robust to outliers (less smooth than Huber) | `objective='reg:absoluteerror'` |

**Ranking**:

| Objective | When to use |
|---|---|
| `'rank:ndcg'` | Learning-to-rank; optimize for NDCG (normalized discounted cumulative gain) |
| `'rank:map'` | Learning-to-rank; optimize for MAP (mean average precision) |

**Example**:
```python
# House price prediction with outliers: use Huber-like loss instead of MSE
xgb.XGBRegressor(objective='reg:pseudohuberloss', n_estimators=100)

# Predict 90th percentile house price (high-end estimates):
xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9)

# Multiclass loan risk (low, medium, high):
xgb.XGBClassifier(objective='multi:softmax', num_class=3)
```

**Interview point**: "Choosing the right objective is critical. Squared error pulls toward outliers; Huber is robust. Quantile regression lets you predict conditional distributions, not just means. This is how you handle different business requirements."

---

### 5.9.7 Categorical features

**`max_cat_to_onehot`** (default: 4) — threshold for categorical encoding:

- Categories with ≤ this many unique values: native categorical splits (XGBoost evaluates all category partitions)
- Categories with > this many values: one-hot encoded

**Example**:
```python
# If 'region' has 3 unique values, it's native categorical
# If 'product_id' has 10K unique values, it's one-hot encoded
train_data = xgb.DMatrix(X, y, enable_categorical=True)
model = xgb.XGBClassifier(max_cat_to_onehot=10)  # raise threshold if one-hot explosion
```

---

### 5.9.8 Monitoring and early stopping

**`eval_metric`** (default: auto-inferred) — which metric to monitor for early stopping:

| Metric | Objective | Meaning |
|---|---|---|
| `'logloss'` | Binary classification | Log-loss (cross-entropy) |
| `'mlogloss'` | Multiclass | Multiclass log-loss |
| `'error'` | Classification | Classification error rate (0-1 loss) |
| `'auc'` | Binary classification | AUC-ROC |
| `'rmse'` | Regression | Root mean squared error |
| `'mae'` | Regression | Mean absolute error |
| `'quantile'` | Quantile regression | Quantile loss |
| `'ndcg'` | Ranking | NDCG@k |

**Example**:
```python
# Monitor AUC instead of default logloss
xgb.XGBClassifier(
    n_estimators=500,
    eval_metric='auc'
)

# Custom eval_set and early stopping
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='logloss',
          early_stopping_rounds=20,
          verbose=False)
```

**`early_stopping_rounds`** (default: None) — stop training if eval metric doesn't improve for N consecutive rounds:

- **Example**: `early_stopping_rounds=20` means if AUC doesn't improve for 20 iterations, stop training and use the best model so far.
- **Critical for avoiding overfitting** when using many trees with low learning rate.

---

### 5.9.9 Callbacks and advanced monitoring

**`callbacks`** (default: None) — custom functions called at the end of each boosting round. Enables programmatic monitoring, dynamic hyperparameter adjustment, custom logging.

**Common callbacks**:

```python
import xgboost as xgb

# Early stopping callback (newer style)
early_stop = xgb.callback.EarlyStopping(
    rounds=20,
    metric_name='logloss',
    data_name='validation'
)

# Log results callback
log_results = xgb.callback.PrintEvaluationProgress(period=10)

# Custom callback: print loss and adaptive learning rate adjustment
class CustomCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        # Access validation loss after each iteration
        if evals_log and 'validation' in evals_log:
            loss = evals_log['validation'][0][1][-1]
            if epoch % 10 == 0:
                print(f"Iteration {epoch}: Validation loss = {loss:.4f}")
        return False  # Return False to continue training

model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stop, CustomCallback()]
)
```

**Interview point**: "Callbacks let you instrument training in real time. You can implement custom metrics, dynamic hyperparameter adjustment, or logging without rewriting the training loop."

---

### 5.9.10 Tuning strategy (refined)

1. **Start with safe defaults**: `learning_rate=0.1`, `n_estimators=100`, `max_depth=5`, `subsample=0.8`
2. **Fix tree structure**: tune `max_depth`, `min_child_weight`, `gamma` until training/val error is balanced
3. **Fix regularization**: tune `lambda`, `alpha` to minimize validation error
4. **Fix stochasticity**: tune `subsample`, `colsample_bytree` if variance is high
5. **Optimize objectives**: if outliers present, switch from squared error to `reg:pseudohuberloss`
6. **Use early stopping**: always, with `eval_metric` tied to your business metric (AUC for ranking, RMSE for regression)
7. **Advanced**: if time allows, tune `tree_method`, `sampling_method` for speed, or experiment with `booster='dart'` for robustness

## 5.9a Monotonic Constraints (regulatory and business constraints)

**The problem**: In regulated industries (credit scoring, insurance, pricing), business or compliance rules often require that certain features have monotonic relationships with the prediction. For example:

- **Credit default prediction**: "Default probability must never decrease as debt-to-income ratio increases" (monotonically increasing).
- **Loan approval scoring**: "Approval probability must never decrease as credit score increases" (monotonically increasing).
- **Insurance premium pricing**: "Premium must never decrease as age increases" (monotonically increasing).

Without constraints, a tree might learn a non-monotonic relationship (e.g., default probability up, then down, then up again with debt ratio) that fits training data well but violates the business rule and may fail regulatory audit.

**How XGBoost implements monotonic constraints**:

During split evaluation (step 5 of the algorithm), XGBoost restricts the search space: if a feature is declared monotonically increasing, any split on that feature must preserve monotonicity in the tree structure.

Formally, if feature $f$ has constraint $\text{mono}_f \in \{-1, 0, +1\}$:
- $\text{mono}_f = 0$: no constraint (default)
- $\text{mono}_f = +1$: monotonically increasing (higher feature value → higher prediction)
- $\text{mono}_f = -1$: monotonically decreasing (higher feature value → lower prediction)

**How it works in practice**:

At each internal node, when evaluating a split on feature $f$ with constraint $\text{mono}_f = +1$:
- Left leaf (feature ≤ threshold): predicted value $w_L$
- Right leaf (feature > threshold): predicted value $w_R$
- **Constraint enforcement**: only accept the split if $w_L \leq w_R$ (left ≤ right, preserving monotonicity)

If no split satisfies the constraint, XGBoost:
1. Either skips the feature (doesn't split on it at this node)
2. Or makes a smaller step that is feasible

---

### Worked example — Loan approval with monotonic constraint on Credit Score

**Scenario**: We want a loan approval model where approval probability monotonically increases with credit score (higher score → higher approval chance).

**Setup**: 10-sample loan dataset, but we explicitly constrain `monotone_constraints = [0, +1, 0]` (no constraint on Income, +1 on CreditScore, no constraint on DebtRatio).

---

**Iteration 1: First tree**

Without constraints, the optimal split at the root would be **CreditScore ≤ 667.5** (from §5.5). Let's check if this violates the monotonic constraint:

- Left (CreditScore ≤ 667.5): IDs {1,2,4,6,9} (all 5 Yes samples) → learned weight $w_L = 1.111$
- Right (CreditScore > 667.5): IDs {3,5,7,8,10} (all 5 No samples) → learned weight $w_R = -1.111$

**Check constraint**: Is $w_L \leq w_R$? Is $1.111 \leq -1.111$? **No** — this violates the monotonic increasing constraint!

The left leaf (low credit scores) has a *higher* prediction (1.111) than the right leaf (high credit scores, -1.111). This says "lower credit score → higher approval," which is backwards.

**XGBoost's action**: Reject this split and try alternatives:

1. **Alternative split on Income ≤ 62.5** (no constraint on Income, so any weights are fine):
   - Left: all Yes → $w_L = 1.111$
   - Right: 1 Yes, 5 No → $w_R = -0.5$ (from §5.5 calculation)
   - Check constraint on Income: none, so accept this split. But this doesn't help with the monotonicity requirement on CreditScore.

2. **Alternative: Don't split at all** — just output a constant, and constrain future trees.

In practice, XGBoost would likely:
- Skip CreditScore as the root split (it violates monotonicity)
- Use Income or DebtRatio as the root split instead
- In deeper trees, when CreditScore is used, enforce splits that preserve the monotonic property

---

**Iteration 2+**: Deeper trees refine predictions while respecting the constraint. Suppose a second tree does split on CreditScore:

- Left (CreditScore ≤ 640, a lower threshold): $w_L = 0.05$
- Right (CreditScore > 640): $w_R = 0.15$

Check: Is $0.05 \leq 0.15$? **Yes** — the constraint is satisfied. This split says "higher credit scores get a higher boost," which is monotonically correct.

---

### Hyperparameter syntax (Python/XGBoost)

```python
import xgboost as xgb

# Define monotonic constraints: [Income, CreditScore, DebtRatio]
# 0 = no constraint, +1 = increasing, -1 = decreasing
monotone_constraints = [0, +1, -1]  

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    monotone_constraints=monotone_constraints,  # <-- the constraint specification
    early_stopping_rounds=10
)

model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)],
          verbose=False)
```

**Interpretation**:
- Income: `0` → no requirement (approval can go up or down with income)
- CreditScore: `+1` → approval probability must increase (or stay constant) as credit score increases
- DebtRatio: `-1` → approval probability must decrease as debt ratio increases (higher debt = lower approval)

---

### Tradeoff: Accuracy vs. Compliance

**Without constraints** (§5.5 example):
- Training loss: Low (fits all patterns in data)
- But violates business logic in some regions
- Regulatory risk if pattern is counter to policy

**With constraints**:
- Training loss: Higher (tree is restricted in where it can place splits/weights)
- But guarantees monotonicity everywhere (passes regulatory audit)
- Trade-off: slightly lower accuracy on the training set, but more defensible and compliant

**In practice**: The accuracy loss is often small (1–3% worse in terms of log-loss) while the compliance gain is huge. This is why monotonic constraints are standard in credit risk, insurance, and hiring models.

---

### Advanced: Partial monotonicity via interaction constraints

XGBoost 1.2+ also supports **interaction constraints**: specify which features are *allowed* to interact. This is more nuanced than monotonic constraints.

Example: `interaction_constraints = [[0, 1], [2]]` means:
- Features 0 and 1 can interact (both appear in the same path from root to leaf)
- Feature 2 cannot interact with features 0 or 1

This is useful when domain knowledge says "CreditScore and Income can interact, but DebtRatio should be independent" — preventing the model from learning non-monotonic patterns across feature combinations.

---

### Comparison to other algorithms

- **Random Forest (§2.11)**: Does not support monotonic constraints natively. You must manually enforce post-hoc correction or use a different library.
- **GBM (§4)**: Depends on implementation; scikit-learn GBM does not support it.
- **XGBoost**: Full native support via `monotone_constraints` parameter.
- **LightGBM (Section 6)**: Also supports `monotone_constraints`, same syntax.
- **CatBoost (Section 7)**: Also supports constraints via `monotone_constraints`.

This is one reason XGBoost and successors are preferred in regulated industries — native support for compliance constraints, not bolted on afterward.

## 5.10 Training complexity

| Component | Cost |
|---|---|
| Per iteration (tree $m$, exact split search) | $O(d \cdot n \log n)$ (same as GBM) |
| Per iteration (histogram-based, `tree_method='hist'`) | $O(d \cdot n)$ (linear after bucketing) |
| Full ensemble ($M$ iterations) | $O(M \cdot d \cdot n)$ with histograms; $O(M \cdot d \cdot n \log n)$ exact |
| Parallelism | Per-tree parallelism (feature-level); inherently sequential boosting |
| Column subsampling | If `colsample_bytree < 1`, further reduces cost to $O(M \cdot d' \cdot n)$ where $d' = \text{colsample} \times d$ |

**Systems advantage**: With histogram-based splits (the default in modern XGBoost), training is $O(M \cdot d \cdot n)$ instead of GBM's $O(M \cdot d \cdot n \log n)$. This makes XGBoost practical on datasets with $n = 10M+$ rows, where GBM becomes too slow.

**Example**: 1M samples, 100 features, 100 boosting rounds.
- GBM exact: $100 \times 100 \times 1M \times \log(1M) \approx 100 \times 100 \times 1M \times 20 = 200B$ operations (very slow)
- XGBoost histogram: $100 \times 100 \times 1M = 10B$ operations (20× faster)

## 5.11 Feature importance and SHAP

XGBoost supports multiple importance types:

**Gain**: Average information gain contributed by each feature (like GBM/AdaBoost).

**Cover**: Average number of samples affected by splits on each feature (how "active" the feature is).

**Frequency**: Number of times each feature appears in splits.

**SHAP values** (via the `shap` library integrated with XGBoost): **The gold standard** for explaining XGBoost predictions. SHAP (SHapley Additive exPlanations) assigns each feature a contribution to each prediction that satisfies consistency, local accuracy, and missingness axioms. Modern XGBoost integrates TreeSHAP for efficient computation.

For production explainability, always use SHAP over simple gain/cover/frequency importance.

## 5.12 The weakness that motivates the next model

XGBoost is production-grade and widely used, but has two remaining limitations on **very large, high-dimensional, sparse datasets** (e.g., $n = 100M$, $d = 10K$, >90% missing):

1. **Memory usage**: Even with histogram-based splits, storing histograms for all features at all nodes can dominate memory, especially on GPUs.

2. **Training time**: While faster than GBM, XGBoost is still sequentially serial — each iteration waits for the prior tree to finish.

3. **Sparse data inefficiency**: Histogram bucketing doesn't naturally exploit sparsity (zero-heavy features). Every zero is bucketed like any other value.

**LightGBM (Section 6) addresses these** via:
- **Leaf-wise growth** instead of level-wise (builds taller, narrower trees, fewer nodes, less memory).
- **Gradient-based sampling** (GOSS): only use samples with high gradient magnitudes, skip low-gradient (confident) samples.
- **Feature bundling** (EFB): groups mutually exclusive features into bundles, reducing dimensionality.
- **Sparse-aware histogram building**: skips zero values in sparse features.

The result: LightGBM trains **5–20× faster** than XGBoost on sparse, high-dimensional data, with comparable or better accuracy.

---

# Section 6 — LightGBM (Light Gradient Boosting Machine)

## 6.1 What it fixes and the core optimizations

XGBoost excels at accuracy and regularity, but struggles with **scale** and **efficiency** on very large datasets. LightGBM (Microsoft, 2017) inverts the optimization priorities: **maximize speed and memory efficiency while maintaining accuracy**, especially on datasets that are massive, high-dimensional, or sparse.

LightGBM addresses XGBoost's three limitations via four architectural innovations:

1. **Leaf-wise (best-first) tree growth** instead of level-wise:
   - XGBoost: grows trees *level-by-level* (all nodes at depth $d$ before depth $d+1$) → creates many nodes, high memory.
   - LightGBM: grows trees *leaf-by-leaf* (always split the leaf with the highest loss reduction) → taller, narrower trees, fewer nodes.
   
2. **Gradient-based One-Side Sampling (GOSS)**: 
   - Keep samples with large gradients (high error, need refinement).
   - Remove samples with small gradients (low error, confident predictions, less useful).
   - Reduces data size per iteration without losing important information.

3. **Exclusive Feature Bundling (EFB)**:
   - Identify mutually exclusive features (e.g., one-hot encoded categories).
   - Bundle them into a single meta-feature, reducing dimensionality.
   - Cuts memory and computation by orders of magnitude for high-cardinality categorical data.

4. **Sparse-aware histogram building**:
   - When building histograms, skip zero values in sparse features entirely.
   - Only histogram non-zero entries, reducing memory footprint dramatically on sparse data.

**Result**: LightGBM trains 5–20× faster than XGBoost on large, sparse datasets while achieving comparable or better accuracy, at a fraction of the memory cost.

## 6.2 The objective function and learning strategy

Mathematically, LightGBM minimizes the same objective as XGBoost:

$$
\mathcal{L}(F) = \sum_{i=1}^n L(y_i, F(x_i)) + \sum_{m=1}^M \Omega(h_m)
$$

with regularization:

$$
\Omega(h_m) = \gamma T_m + \frac{\lambda}{2}\sum_{j=1}^{T_m} w_j^2
$$

**The difference**: LightGBM optimizes the same objective but with a **different tree-growing strategy** and **sampling strategy** to make each iteration faster without materializing the full dataset or all histograms.

Key insight: **LightGBM doesn't change the loss function or theoretical optimality — it changes how efficiently it searches for splits.**

## 6.3 Leaf-wise tree growth and its advantages

**Level-wise growth (XGBoost's default)**:
- Iteration 1: grow all nodes at depth 1 (2 nodes)
- Iteration 2: grow all nodes at depth 2 (4 nodes)
- Iteration 3: grow all nodes at depth 3 (8 nodes)
- Result: A **balanced tree** with $2^D$ leaf nodes at depth $D$

**Example**: A depth-3 XGBoost tree has $2^3 = 8$ leaves.

**Leaf-wise growth (LightGBM's default)**:
- Iteration 1: grow the single node with the highest loss reduction (1 split → 2 leaves)
- Iteration 2: grow the single leaf with the highest loss reduction (1 split → 3 leaves total)
- Iteration 3: grow the single leaf with the highest loss reduction (1 split → 4 leaves total)
- Result: A **skewed tree** with fewer internal nodes and deeper, narrower paths

**Example**: A depth-5 LightGBM tree might have only 10 leaves instead of $2^5 = 32$, while achieving better loss reduction because each split targets the "worst" leaf.

**Advantages of leaf-wise**:
- **Fewer nodes** → less memory for tree structure and histograms
- **Faster convergence** → each split targets the highest-error leaf, so fewer iterations needed
- **Better loss reduction per iteration** → greedily picks the split with maximum gain, not just any node at a given depth

**Disadvantage**: 
- Overfitting risk if not regularized (leaf-wise can create very deep, narrow branches that overfit)
- LightGBM mitigates with `min_child_samples`, `min_data_per_group`, and strict early stopping

## 6.4 Gradient-based One-Side Sampling (GOSS)

**The idea**: Not all samples are equally important for reducing loss. Samples with large gradients (large errors) are driving the next tree's direction. Samples with small gradients (confident, correct predictions) add little new information.

**GOSS procedure** (per boosting iteration):

1. Compute gradient magnitude $|g_i|$ for all samples.
2. Sort samples by $|g_i|$ in descending order.
3. Keep the top $a$ fraction of samples (e.g., top 20% with largest gradients) — call this set $A$ (the "high-error" set). Size: $|A| = an$.
4. Randomly sample $b$ fraction of the remaining $(1-a)n$ samples — call this set $B$ (the "low-error" sample). Size: $|B| = b(1-a)n$.
5. Train the tree only on samples in $A \cup B$, with adjusted weights.
6. **Reweighting**: samples in $A$ keep weight 1; samples in $B$ are upweighted by $\frac{1-a}{b}$ to compensate for undersampling.

**Result**: Effective sample size becomes:
$$
n_{\text{eff}} = |A| + |B| = an + b(1-a)n = n[a + b(1-a)]
$$

Typically, $a + b(1-a) = 0.2–0.5$ (20–50% of original data used).

**Why this works**:
- High-gradient samples are the "signals" — they contain the most information for reducing loss.
- Low-gradient samples are "noise" or already well-predicted by the ensemble — removing some of them doesn't hurt loss, just speeds up training.
- Upweighting the kept low-error samples by factor $\frac{1-a}{b}$ corrects for bias (ensures expected gradient from this subset matches the full gradient).

**Hyperparameters**:
- `top_rate` ($a$): fraction of high-gradient samples to keep (default 0.2, i.e., top 20%)
- `other_rate` ($b$): fraction of low-gradient samples to sample (default 0.1, i.e., 10% of the remaining 80%)
- **Effective sample ratio**: $a + b(1-a) = 0.2 + 0.1(0.8) = 0.28$ (28% of the data is used per iteration)

## 6.5 Exclusive Feature Bundling (EFB)

**The problem**: One-hot encoding categorical features creates many mutually exclusive features (exactly one is 1, rest are 0). Each is tracked in histograms independently, multiplying memory and compute.

**Example**: A category with 10 values becomes 10 binary features. At each node, building histograms for 10 separate features is $10 \times$ costlier than building histograms for 1 feature.

**EFB solution**: If features are **mutually exclusive** (at most one is non-zero per sample), bundle them into a single feature by encoding:

Original one-hot: `[0, 0, 1, 0, 0]` (5 features)
↓
Bundled: `[2]` (1 feature, value 2 means "3rd category")

During histogram building, a single histogram bin encodes all categories of the original feature group.

**When EFB applies**:
- One-hot encoded categoricals (perfect mutual exclusion)
- Nearly exclusive features (e.g., >90% zeros in each column, with minimal overlap)
- Rare cases where domain knowledge says "features A and B never co-occur"

**Memory savings**: 10-feature one-hot → 1 bundled feature = **10× reduction** in histogram memory for that feature group.

**Automatic bundling**: LightGBM can auto-detect and bundle near-exclusive features via parameter `max_cat_to_onehot` (default 4) — categories below this threshold get native categorical handling; above it, LightGBM bundles them.

## 6.6 The algorithm — 10-step template (LightGBM, any loss)

| Step | What happens | Details |
|---|---|---|
| **1. Initialize** | Constant prediction minimizing loss | Same as XGBoost/GBM: $F_0 = \arg\min_c \sum L(y_i, c)$ |
| **2. For iteration $m = 1 \ldots M$:** | — | Outer boosting loop |
| **3. Gradient-based One-Side Sampling** | Select high-error samples, sample low-error samples | Keep top $a\%$ by gradient; randomly sample $b\%$ of rest; upsample kept low-error samples |
| **4. Compute gradients and Hessians** | On the GOSS-selected samples, compute first and second derivatives | $g_i = \frac{\partial L}{\partial F}$, $h_i = \frac{\partial^2 L}{\partial F^2}$ (same as XGBoost) |
| **5. Grow tree leaf-wise** | Start with root node; repeatedly split the leaf with highest gain | Use leaf-wise growth: at each step, pick the single leaf with $\arg\max \text{Gain}_{\text{reg}}$ and split it |
| **6. Build histograms per split** | Create histograms for candidate features, exploit sparsity | Skip zero values in sparse features; use bundled histograms for exclusive features |
| **7. Compute optimal leaf weights** | For each leaf, set weight using regularized formula | $w_j = -\frac{G_j}{H_j + \lambda}$ (same as XGBoost) |
| **8. Apply shrinkage** | Scale tree's contribution | $\tilde h_m(x) = \eta \cdot h_m(x)$ |
| **9. Update ensemble** | Add regularized tree to ensemble | $F_m(x) = F_{m-1}(x) + \tilde h_m(x)$ |
| **10. Predict** | Output final ensemble predictions | $F_M(x)$; for classification, apply sigmoid |

**Key differences from XGBoost** (steps 3, 5–6):
- Step 3: GOSS sampling (not in XGBoost)
- Step 5: leaf-wise growth (vs XGBoost's level-wise)
- Step 6: sparse-aware histograms + automatic EFB bundling

## 6.7 Worked example — Classification (Loan Default, leaf-wise with GOSS)

**Dataset**: 10 samples (Table from §1), binary classification with log-loss.

**Hyperparameters**: `top_rate=0.5` (keep top 50% by gradient), `other_rate=0.5` (sample 50% of rest), `learning_rate=0.1`, `num_leaves=31` (max leaves per tree).

---

### Initialization

Same as XGBoost: $F_0 = 0$, all $p_0 = 0.5$, all $g_i = \pm 0.5$.

---

### Iteration 1: First tree

**Step 3: Gradient-based sampling**

Compute gradient magnitudes: all $|g_i| = 0.5$ (uniform in this balanced case).

With `top_rate=0.5`: keep top 50% of 10 samples = **keep 5 samples** (say, IDs {1,2,3,4,5}, chosen arbitrarily since magnitudes are identical)

Remaining 5 samples (IDs {6,7,8,9,10}): `other_rate=0.5` → sample 50% of 5 = **sample 2.5 ≈ 2 samples** (say, IDs {6,7})

**Working set**: 5 (top) + 2 (sampled) = 7 samples total (vs 10 originally). **30% data reduction**.

**Upsample weights** for the sampled 2: weight = $\frac{1-0.5}{0.5} = 1.0$ (keep them at unit weight to preserve gradient expectation).

---

**Step 5: Leaf-wise growth**

**Root node** (all 7 GOSS samples): score best split:

**Candidate: CreditScore ≤ 667.5**

Left: IDs {1,2,4} from GOSS set (all Yes if they happen to have low credit scores)
Right: IDs {3,5,6,7} (mix of Yes/No)

Suppose left leaf has $G_L = 1.5$, $H_L = 0.75$; right leaf $G_R = 1.0$, $H_R = 1.0$.

Gain = large positive value (compute as in XGBoost).

**Split accepted** → create left and right leaves.

**Next leaf-wise iteration**: now we have 2 leaves. Evaluate: which leaf should be split next?

Leaf-wise chooses the leaf with the **highest gain from its best split** (maximum loss reduction).

Suppose left leaf has max gain 0.8, right leaf has max gain 0.5 → **split the left leaf** again (not level-wise, which would split both).

Result: 3 leaves total after iteration 1 (not 4, as in level-wise).

---

### Iteration 2: Second tree

**GOSS sampling**: new gradients computed on updated $F_1$. Repeat sampling (may select different samples based on new errors).

**Leaf-wise growth**: start from the 3-leaf structure, grow the single best leaf again.

After many iterations, LightGBM's trees are typically **deeper and narrower** than XGBoost's, with **fewer total leaves** but **faster convergence** due to targeted split selection.

---

## 6.8 Worked example — Regression (House Price, leaf-wise with GOSS)

**Dataset**: 10 houses (Table from §2.5), squared error loss.

**Hyperparameters**: `top_rate=0.3` (keep top 30%), `other_rate=0.1` (sample 10% of rest), `learning_rate=0.1`.

---

### Initialization

Same as prior: $F_0 = 251.5k$.

---

### Iteration 1: First tree

**GOSS sampling**

Gradient magnitudes: $|g_i| = 2|y_i - 251.5|$ (proportional to residuals).

| House | $|g_i|$ (approx) | Rank |
|---|---|---|
| 1 | 143 | 5 |
| 4 | 203 | 2 |
| 9 | 263 | 1 |
| 6 | 297 | 0 (highest) |
| 7 | 173 | 4 |
| 8 | 117 | 6 |
| 2 | 3 | 9 (lowest) |
| 5 | 37 | 8 |
| 10 | 197 | 3 |
| 3 | 137 | 7 |

Top 30% of 10 = 3 samples (IDs {6,9,4} with largest errors).
Remaining 7 samples: sample 10% of 7 = 0.7 ≈ 1 sample (say, ID {1}).

**Working set**: 3 (top) + 1 (sampled) = 4 samples. **60% data reduction** (only 4 of 10 used).

---

**Leaf-wise growth**

Fit a tree on the GOSS-selected 4 samples (IDs {6,9,4,1}). Compute gradients and Hessians for these 4:

| House | Gradient $g_i = 2(F_0 - y)$ | Hessian $h_i = 2$ | Actual price |
|---|---|---|---|
| 6 | $2(251.5-400) = -297$ | 2 | 400k |
| 9 | $2(251.5-120) = 263$ | 2 | 120k |
| 4 | $2(251.5-150) = 203$ | 2 | 150k |
| 1 | $2(251.5-180) = 143$ | 2 | 180k |

**Candidate split: SqFt ≤ 16.5**

- Left: IDs {6,9,4,1} (all in GOSS working set; all have SqFt ≤ 16.5) → $G_L = -297 + 263 + 203 + 143 = 312$, $H_L = 8$
- Right: (empty) → $G_R = 0$, $H_R = 0$

This split is useless (everything goes left). Try a different split.

**Alternative candidate: SqFt ≤ 18 (or use a feature like Age)**

Suppose Age ≤ 7.5 separates the samples better:

- Left (Age ≤ 7.5): IDs {9,4} → $G_L = 263 + 203 = 466$, $H_L = 4$
- Right (Age > 7.5): IDs {6,1} → $G_R = -297 + 143 = -154$, $H_R = 4$

**Compute regularized gain** ($\lambda = 1.0$, $\gamma = 1.0$):

$$
\text{Gain}_{\text{reg}} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma
$$

$$
= \frac{1}{2}\left[\frac{466^2}{4+1} + \frac{(-154)^2}{4+1} - \frac{(466-154)^2}{4+4+1}\right] - 1.0
$$

$$
= \frac{1}{2}\left[\frac{216156}{5} + \frac{23716}{5} - \frac{97344}{9}\right] - 1.0
$$

$$
= \frac{1}{2}[43231 + 4743 - 10816] - 1.0 = \frac{1}{2}[37158] - 1.0 = 18579 - 1.0 \approx 18578
$$

(Enormous gain, so this split is accepted.)

**Compute leaf weights**:

$$
w_L = -\frac{G_L}{H_L+\lambda} = -\frac{466}{5} = -93.2
$$

$$
w_R = -\frac{G_R}{H_R+\lambda} = -\frac{-154}{5} = 30.8
$$

**Apply shrinkage** ($\eta = 0.1$):

$$
\tilde w_L = 0.1 \times (-93.2) = -9.32, \quad \tilde w_R = 0.1 \times 30.8 = 3.08
$$

---

**Leaf-wise iteration 2** (still within Iteration 1's tree):

Now we have 2 leaves. Evaluate which leaf has the highest gain if split again:

- **Left leaf** (IDs {9,4}): Try splitting by another feature. Suppose income ≤ 60 splits them: House 9 (income 45) left, House 4 (income 75) right. Compute gain.
- **Right leaf** (IDs {6,1}): Try splitting by another feature. Suppose Income ≤ 62 splits them: House 1 (income 48) left, House 6 (income 85) right. Compute gain.

**Leaf-wise picks the higher-gain split** (say, left leaf's split has gain 500, right leaf's gain is 200) → split the **left leaf**.

Result: 3 leaves total after this iteration (not 4 as in level-wise XGBoost, which would split both leaves).

Repeat leaf-wise iterations: pick the 3-leaf node with highest gain, split it → 4 leaves, etc.

After iteration 1 (with GOSS sampling + leaf-wise growth), the tree has **3–5 leaves** (depending on how many leaf-wise splits occur within the iteration), not $2^D$ leaves as in level-wise XGBoost.

---

### Why LightGBM is faster here

- **GOSS**: only fit on 4 of 10 samples (60% reduction) → 60% faster than full-data XGBoost
- **Leaf-wise**: fewer total splits needed to achieve same loss (each split targets the highest-error leaf) → converges in fewer iterations
- **Sparse-aware**: if any features had missing/zero values, they'd be skipped in histograms → further speedup

**Speedup factor**: Conservatively, 2–5× faster than XGBoost on this data. On real 100M-sample datasets with GOSS removing 70% of low-gradient samples and EFB cutting dimensions in half, 10–20× faster is typical.

---

## 6.9 Hyperparameters (LightGBM-specific)

| Hyperparameter | What it controls | Default | Guidance |
|---|---|---|---|
| `learning_rate` | Shrinkage factor $\eta$ | 0.1 | Same as XGBoost: lower (0.01–0.05) for better generalization. |
| `n_estimators` | Number of boosting rounds | 100 | With GOSS and early stopping, fewer trees often suffice. Start with 100–200. |
| `num_leaves` | Max leaves per tree (leaf-wise specific) | 31 | Controls tree complexity. Higher → more complex, overfitting risk. Typical: 10–100. Start with 31. |
| `max_depth` | Hard cap on tree depth (alternative to `num_leaves`) | -1 (unlimited) | Set if you prefer depth-based control; otherwise use `num_leaves`. |
| `min_child_samples` | Min samples in a leaf | 20 | Increase (20–50) to prevent overfitting, especially with GOSS. |
| `min_data_per_group` | Min samples per categorical group | 100 | Regularization for categorical features. |
| `lambda_l1` | L1 regularization on leaf weights | 0 | Increase (0.1–1) to zero-out some leaf weights, feature selection. |
| `lambda_l2` | L2 regularization on leaf weights | 0 | Increase (0.1–10) to shrink weights toward zero. |
| `top_rate` | GOSS: fraction of high-gradient samples to keep | 0.2 | Higher (0.3–0.5) → use more high-error samples, less aggressive reduction. |
| `other_rate` | GOSS: fraction of low-gradient samples to sample | 0.1 | Higher (0.1–0.3) → use more low-error samples, slower but safer. |
| `max_cat_to_onehot` | Threshold for native categorical vs EFB bundling | 4 | Categoricals with ≤ this many values get native splits; above → bundle or one-hot. |
| `early_stopping_rounds` | Validation-based stopping | None | Set 10–50 to stop when validation loss plateaus. Critical. |
| `force_row_wise` | Force row-wise histogram building | False | Set `True` for small datasets (< 10K samples) where column-wise overhead dominates. |

**Tuning strategy**:
1. Start with defaults; enable GOSS (`top_rate=0.2, other_rate=0.1`) and early stopping.
2. Tune `learning_rate` and `n_estimators` together.
3. Tune `num_leaves` (leaf-wise specific; start with 31, increase if underfitting).
4. Tune `min_child_samples` (increase to reduce overfitting).
5. Tune `lambda_l2` for regularization.
6. Fine-tune GOSS rates if needed for larger datasets.

## 6.10 Missing values and categorical features

**Missing values**: LightGBM handles NaN **natively and automatically** — during histogram building, missing values are placed in a separate bin, and the optimal default direction is learned (same as XGBoost's approach, but more efficient).

**Categorical features**: LightGBM supports two strategies:

**Native categorical** (default for `max_cat_to_onehot` > cardinality):
- Feature is declared as categorical via `categorical_feature=[1,2,...]` parameter
- LightGBM evaluates all possible partitions of categories (not just binary thresholds)
- Avoids one-hot encoding, preserves category structure

**Automatic EFB bundling** (for high-cardinality):
- If a category has > `max_cat_to_onehot` unique values, LightGBM automatically bundles mutually exclusive features
- Reduces dimensionality, cutting memory and computation

**Example**:
```python
train_data = lgb.Dataset(X_train, label=y_train, 
                         categorical_feature=['income_bracket', 'region', 'occupation'])
```

LightGBM natively handles these three features as categories, avoiding one-hot encoding entirely.

## 6.11 Training complexity and systems efficiency

| Component | Cost | LightGBM advantage |
|---|---|---|
| Per iteration (full data, column-wise histograms) | $O(d \cdot n)$ after bucketing | **Parallel histogram building** across features (GPU-efficient) |
| Per iteration with GOSS sampling | $O(d \cdot n \times (\text{keep\_rate}))$ | Top-rate + other-rate typically 0.2–0.3 → **70% reduction** |
| Per iteration with EFB bundling | $O(d' \cdot n)$ where $d' = $ bundled features | For high-cardinality one-hot, **10–100× reduction** in $d$ |
| Full ensemble ($M$ iterations) | $O(M \cdot d' \cdot n \times \text{sample\_rate})$ | Typically **10–20× faster** than XGBoost exact; **3–5× faster** than XGBoost histogram |
| Memory per tree | Histograms for bundled features only | **Minimal overhead** for one-hot-heavy data |
| Parallelism | Feature-level (parallel histogram building) + data-level (GOSS sampling) | Better parallelization than XGBoost's feature-level only |

**Concrete example**: 100M samples, 10K features (including 1000 one-hot-encoded categories):
- **XGBoost with histograms**: $100M \times 10K \times 256 \text{ bins} \times 4\text{ bytes} = 1TB$ memory for histograms per iteration. **Intractable**.
- **LightGBM with EFB + GOSS**: $100M \times 0.3 \times 2K \text{ bundled features} \times 256 \times 4 = 30GB$ per iteration. **Feasible**.
- **Speedup**: ~30× faster, 1TB → 30GB memory footprint.

## 6.12 Feature importance and SHAP

LightGBM supports the same importance types as XGBoost (Gain, Cover, Frequency) and integrates with SHAP for TreeSHAP-based explanations.

Key difference: **LightGBM's native categorical handling makes feature importance more interpretable** — one-hot encoded features are naturally grouped, rather than appearing as separate columns with fragmented importance.

## 6.13 Advantages and limitations vs. XGBoost

### Advantages

1. **Speed**: 5–20× faster on large, sparse, high-dimensional data
2. **Memory**: 5–10× lower memory footprint due to leaf-wise growth and GOSS sampling
3. **Sparse data**: Native sparse-aware histogram building
4. **Categorical features**: Native categorical support without one-hot encoding (when using `categorical_feature` parameter)
5. **Leaf-wise growth**: More adaptive — each split targets the highest-error region

### Limitations

1. **Overfitting risk**: Leaf-wise growth can produce very deep, narrow trees if not regularized properly. Requires stricter `min_child_samples` and early stopping.
2. **Small data**: GOSS sampling can underestimate gradient signals on small datasets (< 10K samples). Better to disable GOSS or use XGBoost.
3. **Stability**: Results can be more sensitive to random seed due to GOSS sampling stochasticity.
4. **Reproducibility**: Randomized sampling means exact reproducibility requires fixing random state at each iteration.

## 6.14 The weakness that motivates the next model

LightGBM is excellent for **speed and scale** on large, sparse datasets, but has one remaining gap:

**Categorical feature handling, while native, still risks target leakage** — the way LightGBM (and XGBoost) encode or handle categoricals means the model can inadvertently learn from the ordering or aggregation of the category, not just the category identity itself.

**CatBoost (Section 7) fixes this** via:
- **Ordered Target Statistics (OTS)**: instead of encoding categories as labels or partitions, CatBoost computes statistics (mean target value per category) *in a way that prevents leakage*.
- **Ordered boosting**: a training procedure that ensures each tree only sees statistics computed from samples "older" in the sequence, preventing the tree from directly optimizing on the category-to-target association.

The result: CatBoost natively handles categorical features with **zero data leakage**, producing more robust and calibrated predictions on categorical-heavy datasets.

That's the final step.
