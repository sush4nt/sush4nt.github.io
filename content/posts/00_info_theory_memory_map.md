---
title: "Information Theory — Memory Map for Tree Models"
date: 2026-07-19T00:00:00+05:30
draft: false
tags: ["information-theory", "tree-models", "interview-prep"]
summary: "A compact memory map connecting entropy, Gini impurity, and information gain to the specific interview questions they answer."
---

# Information Theory — Memory Map for Tree Models

---

## The Organizing Principle (read this first)

All seven concepts below answer one of exactly **three questions**. Lock this in before anything else.

```
┌─────────────────────────────────────────────────────────────────────┐
│ GROUP 1 — "How impure/uncertain is this node?"                      │
│   Single distribution. No split yet. Measuring a node in isolation. │
│   → Shannon Entropy,  Gini Impurity                                 │
├─────────────────────────────────────────────────────────────────────┤
│ GROUP 2 — "Does splitting on X clarify Y?"                          │
│   Two variables: a feature X and a label Y. Evaluating a split.    │
│   → Conditional Entropy,  Information Gain / Mutual Information     │
├─────────────────────────────────────────────────────────────────────┤
│ GROUP 3 — "How wrong is my model q vs. the true distribution p?"    │
│   Two distributions: truth p, model q. Evaluating a model.         │
│   → Cross-Entropy,  KL Divergence                                   │
└─────────────────────────────────────────────────────────────────────┘
                    + one regression bridge concept:
                      Variance Reduction = Group 2 for continuous Y
```

Gini and Entropy answer the *same* Group 1 question — they're not conceptually different, just computationally different.
Conditional Entropy and Information Gain answer the *same* Group 2 question — they're the same split viewed from opposite sides.
Cross-Entropy and KL Divergence answer the *same* Group 3 question — one is the total cost, the other is just the extra cost.

---

## Group 1 — How impure is this node?

### Shannon Entropy

| | |
|---|---|
| **Question** | How uncertain/impure is this single distribution? |
| **Formula** | $H(X) = -\sum_i p_i \log_2 p_i$ |
| **Input** | One distribution (class probabilities in a node) |
| **Output** | Bits of uncertainty. Zero = pure. Max = $\log_2 k$ for $k$ classes |
| **Used in** | ID3, C4.5 — to score node impurity before and after a split |
| **Intuition** | Measures expected surprise. Pure node has zero surprise (you always know the answer). Fully mixed node has maximum surprise (you can't predict anything). |

---

### Gini Impurity

| | |
|---|---|
| **Question** | How uncertain/impure is this single distribution? *(same question as entropy)* |
| **Formula** | $Gini = 1 - \sum_i p_i^2$ |
| **Input** | One distribution (class probabilities in a node) |
| **Output** | Expected misclassification rate. Zero = pure. Max = $0.5$ for binary |
| **Used in** | CART, scikit-learn (default), most production libraries |
| **Intuition** | If you randomly drew a sample and randomly guessed its label using the node's own probabilities, Gini is the probability you'd be wrong. Same shape as entropy, no logarithm, so ~2× cheaper to compute. |

**Why both exist**: Gini is the first-order Taylor approximation of entropy. They rank splits almost identically in practice. Gini won in production because it's faster to compute, not because it's more correct.

---

## Group 2 — Does splitting on X clarify Y?

These two concepts evaluate the *same split* from **opposite directions**. One is the residual, one is the reduction.

```
  Parent entropy H(Y)   =   what REMAINS H(Y|X)   +   what's REMOVED IG(Y,X)
       1.0 bit          =        0.605 bits         +        0.395 bits
```

### Conditional Entropy

| | |
|---|---|
| **Question** | How much uncertainty in Y **remains** after you split on X? |
| **Formula** | $H(Y \mid X) = \sum_x p(x)\, H(Y \mid X=x)$ |
| **Input** | Feature X (the split), label Y |
| **Output** | Weighted average entropy of the child nodes |
| **Used in** | Tree split selection — this is what gets *minimized* to find the best split |
| **Intuition** | After routing samples left and right, how impure are the children on average? Lower is better. A perfect split gives $H(Y|X) = 0$ — the children are completely pure. |

---

### Information Gain = Mutual Information

| | |
|---|---|
| **Question** | How much uncertainty in Y was **removed** by splitting on X? |
| **Formula** | $IG(Y,X) = I(X;Y) = H(Y) - H(Y \mid X)$ |
| **Input** | Feature X (the split), label Y |
| **Output** | Bits of uncertainty eliminated |
| **Used in** | ID3, C4.5 — this is what gets *maximized* to find the best split |
| **Intuition** | The complement of conditional entropy. Since $H(Y)$ is fixed at any given node, maximizing IG is *exactly the same operation* as minimizing conditional entropy — they always select the same split. |

> **"Information Gain" and "Mutual Information" are the same formula**, coined by different communities (ML trees vs. information theory). The tree literature says "maximize information gain." Information theory says "maximize mutual information." Same calculation.

**Key symmetry**: $I(X;Y) = I(Y;X)$. Mutual information is symmetric — knowing X reduces uncertainty about Y by exactly as much as knowing Y reduces uncertainty about X. (Conditional entropy is *not* symmetric: $H(Y|X) \neq H(X|Y)$ in general.)

---

## Group 3 — How wrong is the model vs. truth?

These two also evaluate the *same mismatch* from opposite directions. One is the total encoding cost, the other is just the extra cost above the irreducible floor.

```
  Cross-Entropy H(p,q)  =  True entropy H(p)  +  KL Divergence D_KL(p||q)
       1.004 bits        =     0.971 bits       +     0.033 bits
       (Model A cost)          (floor — can       (waste from using
                                never escape)       wrong model)
```

### Cross-Entropy

| | |
|---|---|
| **Question** | Total bits needed to encode truth $p$ using model $q$ |
| **Formula** | $H(p,q) = -\sum_i p_i \log q_i$ |
| **Input** | True distribution $p$, model distribution $q$ |
| **Output** | Total encoding cost in bits. Always $\geq H(p)$, with equality only when $q = p$ |
| **Used in** | Classification loss function (log-loss). What gradient boosted classifiers minimize over training data. |
| **Intuition** | Uses the true label's probability ($p_i$) to weight how expensive each prediction is, but evaluates the *model's* probability ($\log q_i$) as the cost. Bad predictions (low $q_i$ when $p_i$ is high) are penalized heavily by the $-\log$ term. |

---

### KL Divergence

| | |
|---|---|
| **Question** | Extra bits wasted by using model $q$ instead of the true $p$ |
| **Formula** | $D_{KL}(p \| q) = \sum_i p_i \log \dfrac{p_i}{q_i} = H(p,q) - H(p)$ |
| **Input** | True distribution $p$, model distribution $q$ |
| **Output** | Extra cost above the irreducible floor $H(p)$. Always $\geq 0$, equals $0$ only when $q = p$ |
| **Used in** | Theoretical framing of model optimization; VAEs; generative models |
| **Intuition** | $H(p)$ is the minimum possible cost — the noise floor you can never escape. KL is the "tax" you pay for using the wrong model. A perfect model pays zero tax. |

**The key equivalence** (the one that connects all of this to gradient boosting):

$$
\text{minimize}\ H(p,q) \equiv \text{minimize}\ D_{KL}(p \| q) \equiv \text{maximize likelihood}
$$

Since $H(p)$ is fixed (the true data distribution doesn't change when you update model parameters), minimizing cross-entropy and minimizing KL are the same optimization. This is why log-loss *is* the right classification loss — training on it is equivalent to making your model's predicted probabilities as close as possible to the true ones.

---

## Variance Reduction — The Regression Bridge

| | |
|---|---|
| **Question** | How much does splitting on X reduce spread in continuous Y? |
| **Formula** | $\Delta\text{Var} = \text{Var}(D) - \left(\dfrac{n_L}{n}\text{Var}(D_L) + \dfrac{n_R}{n}\text{Var}(D_R)\right)$ |
| **Input** | Feature X (the split), continuous label Y |
| **Output** | Reduction in weighted variance |
| **Used in** | All regression trees (CART, GBM, XGBoost, LightGBM) |
| **Intuition** | Exact analogue of Information Gain for continuous targets. Same operation — reduce weighted child impurity relative to parent — with variance as the impurity measure instead of entropy. Mathematically equivalent to differential entropy under Gaussian noise assumption. |

---

## The Critical Connections (one view)

```
                          H(Y)
                         ╱    ╲
         what REMAINS ╱        ╲ what was REMOVED
                     ╱          ╲
                H(Y|X)    +    IG(Y,X) = I(X;Y)
           (minimize this)    (maximize this)
           ← same split, opposite directions →


                         H(p,q)
                        ╱      ╲
          irreducible ╱          ╲ model's fault
                    ╱             ╲
                 H(p)      +    D_KL(p||q)
            (can't escape)    (minimize this)
           ← same mismatch, opposite decomposition →


                Group 1 links:
                Gini ≈ H  (first-order Taylor approximation of entropy)
                → same question, Gini is cheaper, rarely different in practice

                Regression bridge:
                ΔVar  ≡  IG  (for continuous targets under Gaussian assumption)
```

---

## Confusing Pairs — Resolved

| Pair | How they differ | Key question to ask yourself |
|---|---|---|
| **Entropy vs. Gini** | Same concept, different formula. Gini approximates entropy. | Is compute speed a concern? If yes, Gini. Rarely matters for accuracy. |
| **Conditional Entropy vs. IG** | Same split, opposite directions. $H(Y|X)$ is the residual; IG is the reduction. | Am I asking "how much is left?" (cond. entropy) or "how much was removed?" (IG)? |
| **IG vs. Mutual Information** | Literally the same formula. Different names from different fields. | Is the paper from ML (IG) or information theory (MI)? Same math either way. |
| **Cross-Entropy vs. KL** | Cross-entropy = floor + KL. KL is just the excess. | Training a model? → minimize cross-entropy. Measuring gap between two distributions theoretically? → KL. |
| **Cross-Entropy vs. Entropy** | Entropy = self-encoding cost (one distribution). Cross-entropy = encoding truth with a model (two distributions). | Is $p = q$? If yes, it's entropy. If $p \neq q$, it's cross-entropy. |

---

## One-Line Anchors (memorize these)

| Concept | One-line anchor |
|---|---|
| Entropy | "How uncertain is this pile of samples right now?" |
| Gini | "Entropy but faster — how often would a random guesser be wrong?" |
| Conditional Entropy | "After the split, how uncertain are the children on average?" |
| Information Gain | "How much uncertainty did the split destroy?" (= parent − children) |
| Mutual Information | "Same as IG. Information theory just calls it that." |
| Cross-Entropy | "Total cost of predicting truth $p$ when your model believes $q$" |
| KL Divergence | "Extra cost above the unavoidable floor — the model's error alone" |
| Variance Reduction | "IG but for regression: how much did the split tighten the target values?" |
