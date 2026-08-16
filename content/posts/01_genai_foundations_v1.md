---
title: "GenAI Foundations: LLMs, Text Processing & Agentic Workflows (v1)"
date: 2026-07-15T00:00:00+05:30
draft: false
tags: ["genai", "llm", "nlp", "interview-prep"]
summary: "Foundational reference on LLM/transformer fundamentals, text processing, evaluation strategies, and production GenAI patterns."
---

# GenAI Foundations: LLMs, Text Processing, & Agentic Workflows

**Date:** July 15, 2026  
**Purpose:** Interview-ready reference for defending LLM fundamentals, evaluation strategies, and production GenAI patterns.

---

## Table of Contents

- [0. Quick Mental Model](#0-quick-mental-model)
- [1. Foundational LLMs: What They Are & Why They Matter](#1-foundational-llms-what-they-are--why-they-matter)
- [2. How LLMs Process Text: The "Rhyming Words for Bat" Example](#2-how-llms-process-text-the-rhyming-words-for-bat-example)
- [3. Transformer Architecture: Deep Dive](#3-transformer-architecture-deep-dive)
  - 3.1 The Problem Transformers Solve
  - 3.2 Self-Attention (Surface → In-Depth)
  - 3.3 Multi-Head Attention
  - 3.4 Residual Connections & Layer Normalization
  - 3.5 Position-wise Feed-Forward Network
  - 3.6 Architecture Variants: Encoder-only, Decoder-only, Encoder-Decoder
  - 3.7 Causal Masking & Autoregressive Generation
  - 3.8 KV Caching & Inference Efficiency
  - 3.9 Interview Tiering: Surface vs. In-Depth Cheat Sheet
- [4. Foundational vs. Fine-tuned Models: When & Why](#4-foundational-vs-fine-tuned-models-when--why)
- [5. Evaluating LLM Responses: Metrics That Matter](#5-evaluating-llm-responses-metrics-that-matter)
- [6. Vector Databases: Why Embeddings Need Their Own Storage](#6-vector-databases-why-embeddings-need-their-own-storage)
- [7. Agentic Workflows: When LLMs Become Agents](#7-agentic-workflows-when-llms-become-agents)
- [8. Interview Narrative: Tying It Together](#8-interview-narrative-tying-it-together)
- [9. Summary Table: Quick Reference](#9-summary-table-quick-reference)
- [References & Resources](#references--resources)

---

## 0. Quick Mental Model

Think of an LLM as a **next-token prediction engine trained at scale**:

1. You give it a prompt (e.g., "give me rhyming words for bat")
2. It breaks your words into tokens
3. Encodes tokens as vectors (embeddings)
4. Runs them through a transformer (learns relationships via attention)
5. Predicts the next token based on all previous context
6. Repeats until it hits a stopping condition (end token, max length, etc.)

**Key insight:** Everything LLMs do—whether generating text, answering questions, or reasoning—boils down to this loop. The power comes from doing it on massive data and using attention to understand long-range dependencies.

---

## 1. Foundational LLMs: What They Are & Why They Matter

### Intuition

A **foundational LLM** is a large language model trained on massive, diverse text corpora (books, web, code, etc.) to predict the next token. It learns general-purpose patterns: grammar, facts, reasoning, code, style, etc.

**Key distinction:**
- **Foundational models** (GPT-3, GPT-4, LLaMA, Claude): Pre-trained on broad data, general-purpose, can do many tasks with prompting alone.
- **Fine-tuned models** (domain-specific): Trained or adapted on narrower datasets, optimized for specific tasks (e.g., medical QA, legal document classification).

### Why Foundational?

Foundational models work because:
1. **Scale unlocks capabilities**: Larger models trained on more data solve tasks they were never explicitly trained for (prompt generalization).
2. **Transfer learning**: Patterns learned from next-token prediction on Wikipedia also help answer questions, generate code, and translate.
3. **Few-shot / zero-shot**: Once a model is large enough, it can adapt to new tasks via in-context learning (giving examples in the prompt).

### When Do You Use Foundational Models?

**Use case: You want general-purpose, flexible AI**
- Open-ended question answering
- Content generation (blogs, emails, summaries)
- Code generation / debugging
- Brainstorming, creative tasks
- Reasoning over diverse knowledge

**When NOT to use:**
- Highly specialized domain with limited labeled data → fine-tune instead
- Real-time systems with strict latency (foundational models are slow) → consider distilled/smaller models
- Privacy-critical: data sent to external API → deploy locally or fine-tune in-house

### Key Equation: Loss During Pre-training

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, \ldots, x_{t-1}; \theta)$$

Where:
- $x_t$ is the token at position $t$
- $P(x_t | \cdots)$ is the model's predicted probability of that token
- $\theta$ are the model weights
- **Training objective**: minimize the negative log likelihood (maximize probability of actual next tokens)

This objective teaches the model to predict well across diverse text. Once trained, you can use it for any text-to-text task via prompting or fine-tuning.

### Worked Example: Foundational vs. Fine-tuned

**Scenario:** You want to classify customer support tickets as "urgent" or "routine."

**Option A: Foundational model (GPT-4) with prompting**
```
Prompt:
"Classify the following support ticket as 'urgent' or 'routine'.
Ticket: 'My account is locked and I can't access it.'
Classification:"

Output: "urgent"
```
- Pros: Works immediately, requires no labeled data, handles edge cases well
- Cons: May be overkill cost-wise, API latency, less control

**Option B: Fine-tune a smaller foundational model (LLaMA 7B)**
```
Training data: 500 labeled support tickets
Fine-tuning: Update weights on classification task
Deployment: Self-hosted, fast inference
```
- Pros: Cheaper, faster at inference, in-house control
- Cons: Requires labeled data, harder to debug, may struggle with edge cases

**Interview answer:** "Depends on your constraints. Start with foundational model + prompting if you have the compute budget and latency tolerance. Fine-tune if you need cost control, inference speed, or data privacy."

---

## 2. How LLMs Process Text: The "Rhyming Words for Bat" Example

### The Full Flow

Let's trace what happens when you input: **"Give me rhyming words for bat"**

#### Step 1: Tokenization
The model breaks text into tokens (subword chunks):
```
Input: "Give me rhyming words for bat"
Tokens: ["Give", "me", "rhyming", "words", "for", "bat"]
Token IDs: [1045, 477, 35596, 2356, 329, 9994]  (example IDs)
```

**Why tokens, not characters?**
- Efficiency: fewer tokens = faster processing
- Semantic grouping: "playing" is one token, not 7 characters
- Language structure: punctuation, special chars handled naturally

**Popular tokenizers:** BPE (Byte Pair Encoding), SentencePiece, WordPiece

#### Step 2: Embedding (Lookup)
Each token ID becomes a **dense vector** (embedding):
```
Token ID 1045 ("Give") → Vector [0.23, -0.51, 0.12, 0.09, ...] (768 dims for GPT)
Token ID 477 ("me")   → Vector [-0.10, 0.34, -0.22, 0.56, ...] (768 dims)
...
Token ID 9994 ("bat") → Vector [0.45, 0.02, -0.31, 0.18, ...] (768 dims)
```

**Embedding intuition:** Vectors capture semantic meaning. "bat" and "cat" have similar vectors (rhyme, animal/object). "give" and "provide" are close (synonyms).

**Key formula:**
$$\text{embedding}(token\_id) = E[token\_id]$$

Where $E$ is a learned embedding matrix (vocabulary size × embedding dimension).

#### Step 3: Positional Encoding
Add information about word order (transformers don't inherently know position):
```
Position 0: "Give" embedding + [0.0, 1.0, 0.0, 0.0, ...]
Position 1: "me" embedding + [0.84, 0.54, 0.0, 0.0, ...]
Position 2: "rhyming" embedding + [0.91, -0.42, 0.0, 0.0, ...]
...
Position 5: "bat" embedding + [0.28, -0.96, 0.0, 0.0, ...]
```

**Formula (sinusoidal):**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Why?** Position information tells the model "bat" is at the end, so it rhymes with words we need to generate.

#### Step 4: Transformer Attention (Core Intelligence)
The transformer stack (12–96 layers, depending on model size) runs each embedded token through **multi-head attention** and **feed-forward networks**.

**Attention intuition:** "Which tokens should I focus on to understand this one?"

For the token "bat":
- Attention weights might be: "rhyming" (0.6), "words" (0.25), "for" (0.10), "bat" (0.05)
- This tells the model: "To generate rhymes, pay most attention to the word 'rhyming' and the target word 'bat'."

**Attention formula (simplified):**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = Query (current token)
- $K$ = Keys (all tokens)
- $V$ = Values (embeddings to aggregate)
- $d_k$ = scaling factor (prevents gradient explosion)

**What this does:** Computes relevance of each token to the current token, then takes a weighted average of their values. Multi-head attention repeats this process with different subspaces (e.g., 8 heads × 96 dims each).

After attention, a feed-forward network refines the representation:
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

(ReLU activation, similar to deep learning)

#### Step 5: Decoding (Token Generation)
After transformer layers, the model outputs a probability distribution over the vocabulary (~50k tokens):

```
Softmax output:
P("cat") = 0.25
P("rat") = 0.20
P("hat") = 0.15
P("mat") = 0.12
P("sat") = 0.08
... (rest < 0.05)
```

**How does it pick the next token?**

**Greedy decoding:** Always pick the highest-probability token → deterministic, can get stuck in loops
```
Output: "cat" (0.25 is highest)
```

**Beam search:** Keep the top-K most likely sequences, expand each → better quality but slower
```
Keep top-2 sequences:
1. "The rhyming words for bat are cat..." (cumulative prob: 0.25 × ...)
2. "The rhyming words for bat are rat..." (cumulative prob: 0.20 × ...)
Then expand both forward
```

**Sampling:** Sample from the distribution (stochastic) → more diverse output
```
Sample from P(·) with temperature τ
Higher τ = flatter distribution = more randomness
Lower τ = sharper distribution = more confident
```

#### Step 6: Repeat Until Stopping
The generated token becomes input for the next prediction:
```
Input: "Give me rhyming words for bat cat"
...repeat Steps 1–5...
Next token: "and"

Input: "Give me rhyming words for bat cat and"
Next token: "hat"

... until token = <END> or max_length reached
```

### Full Output Example
```
Input: "Give me rhyming words for bat"
Output: "Give me rhyming words for bat: cat, rat, hat, mat, sat, fat, vat."
```

---

## 3. Transformer Architecture: Deep Dive

**Why this section exists:** Section 2 walked through attention *in the flow* of generating text. This section pulls the transformer apart as a standalone architecture — because interviewers probe it two ways: **surface** ("what is self-attention?") and **in-depth** ("why divide by √d_k? why do we need residual connections? what breaks if you remove layer norm?"). Each concept below is written with a **Surface** line you can say in 10 seconds, and a **In-Depth** block you use if they push further. This mirrors how the questioning actually escalates in an interview.

### 3.1 The Problem Transformers Solve

**Surface:** Before transformers, RNNs/LSTMs processed text one token at a time in sequence — slow to train (no parallelism) and bad at long-range dependencies (vanishing gradients over long sentences). Transformers process **all tokens simultaneously** and use attention to model relationships between *any* two tokens directly, regardless of distance.

**In-Depth:**
- RNNs: token $t$'s representation depends on token $t-1$'s output → inherently sequential → can't parallelize across time steps during training.
- Long-range dependency problem: information from token 1 has to "survive" being passed through 50 sequential updates to reach token 50. Gradients shrink (vanish) or blow up over that path.
- Transformers replace sequential recurrence with **attention**: every token directly looks at every other token in a single step (O(1) path length between any two tokens, vs O(n) for RNNs). This is the single biggest architectural idea — distance in the sequence no longer means distance in computation graph.
- Trade-off: attention is O(n²) in sequence length (every token attends to every other), which is why context-length scaling is expensive and a major production/cost lever (relevant to your JD's MLOps angle).

### 3.2 Self-Attention (Surface → In-Depth)

**Surface:** Self-attention lets each token look at every other token in the sequence and decide how much to "borrow" from each one to build its own contextual representation. It's a weighted average, where the weights are learned relevance scores.

**In-Depth — the mechanics:**

Every token's embedding is projected into three separate vectors via learned weight matrices:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

- **Query (Q):** "What am I looking for?" — represents the current token's request for information.
- **Key (K):** "What do I contain?" — represents what each token offers, to be matched against queries.
- **Value (V):** "What do I actually contribute?" — the content that gets aggregated once relevance is decided.

**Why three separate projections and not just reuse the embedding directly?** Because "how relevant is token A to token B" (Q·K matching) is a different function than "what content should flow from A to B" (V). Separating them gives the model more expressive power — it can learn to attend to one thing while extracting a different thing.

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Breaking this down term by term:
1. $QK^T$ — dot product between every query and every key → a raw relevance score matrix (n × n).
2. $\sqrt{d_k}$ — **scaling factor.** Without it, as $d_k$ (dimension of keys) grows, dot products grow large in magnitude, pushing softmax into a saturated regime where gradients vanish. Dividing by $\sqrt{d_k}$ keeps the variance of the dot products roughly constant regardless of dimensionality. **This is a classic in-depth question: "why √d_k specifically?" — because for random Q,K with unit variance, $Q \cdot K$ has variance $d_k$; dividing by $\sqrt{d_k}$ normalizes it back to unit variance.**
3. $\text{softmax}(\cdot)$ — converts raw scores into a probability distribution (weights sum to 1) so the output is a proper weighted average.
4. $\cdot V$ — weighted sum of value vectors using those probabilities.

**Interview-ready one-liner:** "Self-attention computes, for every token, a weighted average of all other tokens' value vectors — where the weights come from how well that token's query matches every other token's key, scaled to keep gradients stable."

### 3.3 Multi-Head Attention

**Surface:** Instead of computing attention once, the model computes it **multiple times in parallel** ("heads"), each with its own learned Q/K/V projections, then concatenates the results. Different heads learn to specialize in different types of relationships (e.g., one head tracks syntactic dependencies, another tracks coreference).

**In-Depth:**
- If model dimension $d_{model} = 768$ and you use 12 heads, each head operates in a $768/12 = 64$-dimensional subspace. This keeps total compute roughly the same as one big attention operation, while giving the model multiple independent "views."
- Formally: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$, where each $\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$.
- **Why not just use one head with the full dimension?** Empirically, splitting into multiple smaller subspaces lets different heads specialize (visualized in papers: some heads attend to adjacent tokens, some to the subject-of-verb, some to rare long-range dependencies). One large head tends to average all these patterns together and lose that specialization.
- **Common follow-up:** "What happens if you have too many heads?" — each head's subspace shrinks, reducing its capacity to represent a meaningful relationship; there's a practical sweet spot tied to $d_{model}$.

### 3.4 Residual Connections & Layer Normalization

**Surface:** Each transformer sub-layer (attention, then feed-forward) wraps its output with a **residual connection** (add the input back) and a **layer normalization** step. This is what makes it possible to stack 12–96+ layers without training collapsing.

**In-Depth:**
- **Residual connection:** $\text{output} = x + \text{SubLayer}(x)$. This gives gradients a direct path back to earlier layers during backpropagation (an unimpeded "gradient highway"), which is exactly the same problem residuals solved in ResNets for CNNs. Without it, very deep networks suffer vanishing gradients and are hard to optimize.
- **Layer normalization:** Normalizes activations across the feature dimension (not the batch dimension, unlike BatchNorm) so each token's representation has stable mean/variance before being passed to the next layer. This stabilizes training dynamics, especially important since transformer inputs can vary wildly in scale across layers.
- **Pre-LN vs Post-LN (a genuinely advanced/senior-level distinction):** Original transformer applied LayerNorm *after* the residual add ("Post-LN"). Most modern LLMs (GPT-2 onward) apply LayerNorm *before* the sub-layer ("Pre-LN": $x + \text{SubLayer}(\text{LayerNorm}(x))$). Pre-LN gives more stable gradients early in training and allows training much deeper networks without warmup tricks — this is a good "I know the recent literature" signal if asked to go deep.

### 3.5 Position-wise Feed-Forward Network

**Surface:** After attention mixes information *across* tokens, each token independently passes through a small 2-layer MLP that transforms its representation. Attention mixes information between positions; the FFN adds per-token non-linear processing capacity.

**In-Depth:**
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
- Typically expands the dimension 4x in the hidden layer (e.g., 768 → 3072 → 768), giving the model more capacity to transform each token's representation before passing to the next layer.
- **Why is this needed if attention already mixes context?** Attention is fundamentally a *linear* re-weighting/averaging operation (softmax weights times a linear combination of values). Without a non-linear FFN afterward, the whole stack would collapse into something much closer to a linear model, unable to represent complex functions. The FFN is where most of the "reasoning capacity"/parameter count actually lives — in most LLMs, the FFN accounts for roughly two-thirds of total parameters.

### 3.6 Architecture Variants: Encoder-only, Decoder-only, Encoder-Decoder

This is a **very common senior-level question** ("why is GPT decoder-only and BERT encoder-only? Why does T5 use both?") — know it cold.

| Variant | Attention Pattern | Examples | Best For |
|---|---|---|---|
| **Encoder-only** | Bidirectional (every token sees every other token, past and future) | BERT, RoBERTa | Understanding tasks: classification, NER, embeddings — no generation |
| **Decoder-only** | Causal/masked (token $t$ only sees tokens $1..t$) | GPT, LLaMA, Claude | Text generation — the dominant architecture for modern LLMs and chat/agentic systems |
| **Encoder-Decoder** | Encoder is bidirectional; decoder is causal + attends to encoder output ("cross-attention") | T5, BART, original Transformer (translation) | Sequence-to-sequence tasks with a clear input/output split: translation, summarization |

**Why decoder-only dominates modern LLMs (a genuinely good talking point):** Decoder-only models can be trained with a single, simple objective (next-token prediction) on any text, which scales trivially to internet-size corpora. Encoder-decoder architectures need paired input/output data (harder to source at scale) and add complexity (two stacks, cross-attention) for a benefit that mostly matters for narrow seq2seq tasks. Once decoder-only models got large enough, they could do translation/summarization *too* via prompting — removing most of the encoder-decoder's advantage.

### 3.7 Causal Masking & Autoregressive Generation

**Surface:** In decoder-only models, each token is only allowed to attend to itself and earlier tokens — never future ones. This is what makes generation valid: the model can't "cheat" by looking ahead at the answer it's supposed to produce.

**In-Depth:**
- Implemented by adding a mask to the attention score matrix *before* softmax: set all "future" positions to $-\infty$, so after softmax their weight becomes exactly 0.
- This is what lets training be efficient despite generation being sequential at inference time: during training, you can compute the loss for predicting *every* position in a sequence in a single forward pass (since each position's attention is already restricted to its own past), rather than needing one forward pass per token. This parallel-training / sequential-inference asymmetry is a great answer to "how is training different from inference for LLMs?"

### 3.8 KV Caching & Inference Efficiency

**Why this belongs here:** Your JD explicitly calls out end-to-end MLOps ownership and production deployment — this is the concept that separates "I understand transformers academically" from "I understand what makes them expensive/slow to serve," which is exactly the kind of production-engineering depth senior interviewers probe for.

**Surface:** During generation, the model produces one token at a time, and naively would recompute attention over the *entire* growing sequence at every step. KV caching stores the Key and Value vectors for all previous tokens so each new step only computes attention for the *one new* token, reusing cached K/V for everything before it.

**In-Depth:**
- Without caching: generating token $n$ requires recomputing K, V for all $n-1$ previous tokens again — wasteful, since those don't change (causal masking guarantees past tokens' representations are fixed once computed).
- With caching: each generation step only computes $Q, K, V$ for the new token, appends its $K, V$ to the cache, and computes attention of the new $Q$ against the full cached $K, V$ matrix.
- **Cost implication (a good production talking point):** KV cache size grows linearly with sequence length and batch size, and is a major memory bottleneck in serving long-context requests — this is why techniques like multi-query attention (MQA) and grouped-query attention (GQA) exist: they share K/V across multiple heads to shrink cache size, trading a small amount of quality for significantly cheaper serving. If asked about serving cost/latency trade-offs for LLMs in production, this is the concrete mechanism to point to.

### 3.9 Interview Tiering: Surface vs. In-Depth Cheat Sheet

| Question | Surface Answer (10 sec) | If Pushed Further |
|---|---|---|
| What is self-attention? | Weighted average of all tokens' values, weighted by query-key similarity | Explain Q/K/V roles, softmax, and why scale by √d_k |
| Why multi-head? | Multiple parallel attention "views" specializing in different relationships | Subspace dimension math, specialization evidence, too-many-heads trade-off |
| Why residual connections? | Lets gradients flow through deep stacks without vanishing | Compare to ResNets; explain Pre-LN vs Post-LN stability difference |
| Why layer norm, not batch norm? | Normalizes per-token, works with variable-length sequences and small/uneven batches | Explain why BatchNorm's batch-dimension statistics break down for sequence data |
| Why do we need the FFN if we have attention? | Attention mixes context; FFN adds non-linear per-token processing | Explain attention is linear-ish (softmax-weighted linear combo); FFN holds most parameters |
| Encoder vs decoder vs both? | Encoder=understanding (bidirectional), decoder=generation (causal), both=seq2seq | Explain why decoder-only won for scaling LLMs (single objective, more data) |
| Why is inference slower than training feels like it should be? | Generation is sequential (one token at a time) | Explain causal masking enables parallel training but forces sequential decoding; introduce KV cache as the mitigation |
| How would you reduce serving cost for long-context LLMs? | Caching, batching | Introduce MQA/GQA, KV cache memory scaling, quantization |

---

## 4. Foundational vs. Fine-tuned Models: When & Why

### Side-by-Side Comparison

| Dimension | Foundational | Fine-tuned |
|-----------|-------------|-----------|
| **Pre-training** | Broad, diverse corpus (Wikipedia, Books, Web, Code) | Already pre-trained; adapted on task-specific data |
| **Use case** | Open-ended, zero-shot, few-shot tasks | Domain-specific tasks, higher accuracy for narrow use cases |
| **Labeled data required** | No (unsupervised pre-training) | Yes (task-specific labels) |
| **Inference cost** | High (large model, many layers) | Medium (same size, but faster if distilled) |
| **Latency** | High (~1–10 sec for long generations) | Depends on size; can be low if distilled |
| **Customization** | Limited to prompting | Full model retraining |
| **Failure modes** | Hallucination, knowledge cutoff, prompt sensitivity | Overfitting (if limited data), catastrophic forgetting |

### Decision Tree: When to Use Which

```
Task: You need an AI system

├─ Have lots of labeled domain data (1000+)?
│  └─ Yes → Fine-tune. Invest in labeled data, optimize for your distribution.
│  └─ No → Use foundational + prompting
│
├─ Critical latency constraint (<100ms)?
│  └─ Yes → Distill foundational model or use small fine-tuned model
│  └─ No → Use full foundational model
│
├─ Data privacy critical (can't send to API)?
│  └─ Yes → Self-host foundational or fine-tuned model locally
│  └─ No → Use API-based foundational model (easier ops)
│
└─ Need highest accuracy on your specific domain?
   └─ Yes → Fine-tune on domain data (e.g., medical LLaMA on medical texts)
   └─ No → Prompt-engineer a foundational model
```

### Worked Example: Email Classification

**Scenario:** Classify customer emails as "billing," "technical support," "sales," or "general inquiry."

**Option 1: Foundational Model (GPT-4) with Few-Shot Prompting**
```
Prompt:
"Classify the email below into one of: billing, technical support, sales, general.

Examples:
Email: 'I was charged twice for order #1234'
Category: billing

Email: 'My app keeps crashing on startup'
Category: technical support

Email: 'Do you offer enterprise plans?'
Category: sales

Email to classify:
'Hi, just checking in on my recent purchase'
Category:"

Response: "general"
```

**Pros:**
- Zero labeled data needed
- Handles edge cases (model knows nuance)
- No fine-tuning overhead

**Cons:**
- API calls are expensive at scale (100k emails/day = $$$)
- Latency (SLAs may require <100ms)
- Data goes to external API

**Option 2: Fine-tune LLaMA 7B on 500 labeled emails**
```
Training:
- Collect 500 labeled customer emails
- Fine-tune LLaMA 7B on classification task
- Evaluate on 100 held-out test examples
- Deploy on Kubernetes

Inference:
- Single API call to self-hosted endpoint
- ~50ms latency, <$0.001 per classification
```

**Pros:**
- Cheap, fast inference
- Full control, no external API
- Can iterate quickly (fine-tune is fast)

**Cons:**
- Need labeled data upfront
- Model may struggle with out-of-distribution emails (poor generalization)
- Lower accuracy than GPT-4 on edge cases

**Interview answer:** "For 100k emails/month, I'd start with fine-tuned LLaMA 7B—lower cost and latency. If accuracy is critical and budget allows, use GPT-4 for those; use fine-tuned for high-volume, straightforward cases. Hybrid approach."

---

## 5. Evaluating LLM Responses: Metrics That Matter

### The Challenge

Unlike classification (simple: is the prediction correct?), LLM evaluation is hard because:
- **Open-ended tasks** (generation, summarization, QA) have multiple correct answers
- **Quality is subjective** (is this summary good? is this response helpful?)
- **Human eval is slow/expensive**

### Metrics Overview

#### 4.1 Automatic Metrics (No Human Required)

**For Text Generation (e.g., Translation, Summarization):**

**BLEU (Bilingual Evaluation Understudy)**
- **What it does:** Compares generated text to reference(s) using n-gram overlap
- **Formula:** 
$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$
where $p_n$ = precision of n-grams, $BP$ = brevity penalty

- **Intuition:** How many words/phrases from the reference appear in the generated text?
- **Range:** 0–1 (higher is better)
- **Example:**
  - Reference: "The cat sat on the mat"
  - Generated: "The cat sat on a mat"
  - 1-gram match: 6/6 = 1.0 (all words present)
  - 2-gram match: 4/5 (missing "on the")
  - **BLEU ≈ 0.85**

- **Pros:** Fast, no human required
- **Cons:** Doesn't capture meaning, penalizes paraphrases, unreliable for short texts

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- **What it does:** Recall-based comparison (inverse of BLEU)
- **Variants:**
  - ROUGE-N: n-gram overlap (like BLEU but recall-focused)
  - ROUGE-L: Longest common subsequence (cares about word order)
- **Intuition:** How much of the reference is captured in the generation?
- **Better for:** Summarization (cares about not missing key info)
- **Example:**
  - Reference: "The quick brown fox jumps over the lazy dog"
  - Generated: "A fast brown fox leaps over a lazy dog"
  - ROUGE-1 recall: 7/9 ≈ 0.78 (captured 7 of 9 words)

**METEOR**
- **What it does:** Combines precision & recall with synonymy/stemming
- **Intuition:** "The fox leaps" should be similar to "The dog jumps" (synonyms matter)
- **Useful for:** Tasks where paraphrases are acceptable
- **Con:** Slower to compute, requires external alignment tools

**Perplexity**
- **What it does:** Inverse probability assigned to held-out test data
$$\text{Perplexity} = 2^{-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i)}$$
- **Intuition:** How surprised is the model at real data? Lower = model thinks data is likely = better fit
- **Use case:** Language modeling, model comparison (not task-specific)
- **Con:** Doesn't measure usefulness for downstream task

---

#### 4.2 Human Evaluation (Gold Standard)

**When to use:** High-stakes decisions, evaluating quality on open-ended generation

**Typical rubric (1–5 scale):**
- **Relevance:** Does the response answer the question?
- **Factuality:** Is the information correct?
- **Coherence:** Is it well-written and logical?
- **Helpfulness:** Would a user find this useful?

**Example annotation:**
```
Prompt: "Summarize this article in 2 sentences"
Generated summary: "..."
Annotator 1 rating: 4/5 (good summary, one detail missing)
Annotator 2 rating: 5/5 (excellent)
Inter-annotator agreement (Cohen's kappa): 0.72 (fair)
Average score: 4.5/5
```

**Cost:** ~$5–10 per sample (depends on task complexity and annotation platform)

---

#### 4.3 LLM-as-Judge (Emerging, Practical)

Use a strong LLM (GPT-4, Claude) to evaluate other models.

**Prompt:**
```
You are an expert evaluator. Rate the quality of this generated response.

Question: "What is photosynthesis?"
Generated response: "Photosynthesis is a process where plants convert sunlight into chemical energy using chlorophyll."
Reference: "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose."

Rate on accuracy (1-5), completeness (1-5), clarity (1-5).
Provide reasoning.
```

**Output:**
```
Accuracy: 5/5 (correct fundamental explanation)
Completeness: 3/5 (missing detail on glucose production)
Clarity: 5/5 (simple, understandable)
Overall: 4/5
```

**Pros:**
- Fast, cheap (one API call per sample)
- Flexible (can evaluate any task)
- Correlates well with human judgment (empirically validated)

**Cons:**
- Not fully independent (LLM bias may favor LLM-generated style)
- Best used with strong model (GPT-4 > GPT-3.5)

---

### Metric Selection by Task

| Task | Primary Metric | Secondary |
|------|----------------|-----------|
| Translation | BLEU or METEOR | Human eval on sample |
| Summarization | ROUGE-L | Human eval on factuality |
| Question Answering | Exact match (if short answers) or F1 (token overlap) | LLM-as-judge, human eval |
| Open-ended generation | LLM-as-judge or human eval | Perplexity (sanity check) |
| Dialogue/Chat | Human eval only | LLM-as-judge if budget-constrained |

---

### Worked Example: Evaluating a Customer Service Chatbot

**Task:** Generate helpful, accurate responses to customer questions.

**Question:** "Can I return a purchase after 30 days?"

**Reference (gold standard):** "Our return policy allows returns within 30 days of purchase. After 30 days, returns are not accepted unless the product is defective."

**Model A response:** "Yes, we accept returns within 30 days."
- Accuracy: ✓ Correct
- Completeness: ✗ Missing info (no mention of defects)
- BLEU: 0.60 (overlap is low, different words)
- Human rating: 3/5 (helpful but incomplete)
- LLM-as-judge: "Accurate but lacks important condition about defects. 3/5"

**Model B response:** "You can return items if they're broken. We usually accept returns up to 30 days, sometimes longer depending on the situation."
- Accuracy: ✗ Misleading (doesn't clearly state 30-day limit; "usually" is vague)
- Completeness: ~ Partial (mentions defects but unclear on timing)
- BLEU: 0.45
- Human rating: 2/5 (confusing, inaccurate)
- LLM-as-judge: "Vague and potentially misleading about return window. 2/5"

**Verdict:** Model A is better. BLEU and human eval agree.

---

## 6. Vector Databases: Why Embeddings Need Their Own Storage

### Intuition

A **vector database** is optimized for storing, indexing, and searching **high-dimensional vectors** (embeddings). It answers: "Which vectors are most similar to this query vector?"

**Why not use a regular SQL database?**
- SQL: Built for exact matches (`WHERE customer_id = 123`) and range queries (`WHERE price > $50`)
- Vector DB: Built for approximate nearest-neighbor search (`Find the 5 most similar vectors`)

SQL is not designed for similarity in 768-dimensional space.

### How They Work

**Example: Semantic Search on Customer Support Tickets**

**Step 1: Embed the knowledge base**
```
Document 1: "How do I reset my password?"
Embedding: [0.23, -0.51, 0.12, ..., 0.09] (768 dims)

Document 2: "I forgot my account password"
Embedding: [0.24, -0.50, 0.13, ..., 0.08] (768 dims)

Document 3: "What are your shipping rates?"
Embedding: [0.01, 0.15, -0.72, ..., 0.33] (768 dims)

... store all in vector DB with fast indexing
```

**Step 2: Embed the query**
```
User query: "How do I change my password?"
Embedding: [0.25, -0.49, 0.11, ..., 0.10] (768 dims)
```

**Step 3: Find nearest neighbors**
Vector DB computes similarity (e.g., cosine distance) to all documents:
```
Similarity(query, Doc1) = 0.987 ← Highest (most similar)
Similarity(query, Doc2) = 0.985 ← Second
Similarity(query, Doc3) = 0.102 ← Not similar
```

**Step 4: Return top-K results**
```
Top-1: "How do I reset my password?" (similarity: 0.987)
Top-2: "I forgot my account password" (similarity: 0.985)
```

**Why this is better than keyword search:**
- Keyword search: "password" matches both Docs 1–3. Not smart.
- Vector search: Understands that "change password" ≈ "reset password" ≈ "forgot password" semantically.

### Vector DB vs. Relational DB

| Feature | Relational DB (SQL) | Vector DB |
|---------|-------------------|-----------|
| **Data type** | Structured tables (rows, cols) | High-dimensional vectors |
| **Query type** | Exact/range match (WHERE clause) | Similarity/KNN (find closest N) |
| **Indexing** | B-tree, Hash, etc. | HNSW, IVF, LSH (specialized for vectors) |
| **Latency** | Fast exact match; slow for similarity | Fast similarity search |
| **Memory** | Lower (for tabular data) | Higher (vectors are dense) |
| **Examples** | PostgreSQL, MySQL | Pinecone, Weaviate, Milvus, FAISS |

### Vector DB Use Cases

1. **Retrieval-Augmented Generation (RAG)**
   - Embed user question
   - Find relevant documents from vector DB
   - Pass retrieved docs + question to LLM for answering
   - **Why:** Reduces hallucination, adds domain knowledge

2. **Semantic Search**
   - Find documents similar in meaning (not keywords)
   - Example: "Best budget laptop" matches "Cheap computer" even without keyword overlap

3. **Recommendation Systems**
   - Embed user preferences and items
   - Find most similar items to user's interests

4. **Duplicate Detection**
   - Embed documents/emails
   - Find near-identical or very similar items

5. **Image/Audio Search**
   - Embed images/audio as vectors
   - Search "similar images" by visual content (not metadata)

### Worked Example: RAG for Customer Support

**Traditional chatbot approach:**
```
User: "How do I cancel my subscription?"
Chatbot: [searches canned responses or keyword database]
Output: Generic response, may not match their specific question
```

**Vector DB + RAG approach:**
```
Step 1: Offline (setup)
- Embed all help documentation into vector DB
- Documents: "Subscription management guide", "Cancellation policy", etc.

Step 2: Online (user query)
- User: "How do I cancel my subscription?"
- Embed query: [0.12, 0.34, -0.51, ...]
- Vector DB returns top-3 most similar docs:
  1. "How to cancel subscription" (similarity: 0.96)
  2. "Subscription management guide" (similarity: 0.89)
  3. "Refund policy" (similarity: 0.76)

Step 3: Generate response
Prompt to LLM:
"Based on the following documentation, answer the user's question.
Docs:
---
[Top 3 docs retrieved from vector DB]
---
User question: 'How do I cancel my subscription?'
Answer:"

LLM response: "To cancel, go to Settings > Subscription > Click Cancel. You'll receive confirmation via email. Refunds are issued within 5–7 business days."
```

**Benefits:**
- ✓ Grounded in actual documentation (less hallucination)
- ✓ Always up-to-date (docs update → automatically used)
- ✓ Explainable (can show which docs were used)

---

## 7. Agentic Workflows: When LLMs Become Agents

### Intuition

An **agent** is an LLM that can **think, plan, and act** using external tools.

Instead of just generating text, it:
1. **Thinks**: Reasons about the problem
2. **Plans**: Decides what action to take
3. **Acts**: Calls a tool (search, calculator, API, database query)
4. **Observes**: Sees the result
5. **Repeats**: Uses the result to plan the next action

**Key insight:** The LLM is no longer just a text generator—it's an orchestrator that decides what to do.

### Simple Flow: ReAct (Reasoning + Acting)

```
User: "What is the capital of France? What year did it become the capital?"

Agent loop:
1. Think: "I need to find the capital of France and when it became the capital."
2. Act: Call tool [search("capital of France")]
3. Observe: "Paris is the capital of France"
4. Think: "Good, now I need the year it became capital"
5. Act: Call tool [search("when did Paris become capital of France")]
6. Observe: "Paris became the capital in 1528 (moved from Tours)"
7. Think: "I have both pieces of info. I can answer now."
8. Respond: "Paris is the capital of France. It became the capital in 1528."
```

**Compare to non-agentic:**
```
User: "What is the capital of France? What year did it become the capital?"
Non-agentic LLM: "Paris is the capital. I think it became capital in 1589" (hallucination, no guarantee of accuracy)
```

### Core Components

#### 1. Language Model (the "brain")
Decides what to do at each step. Examples: GPT-4, Claude, LLaMA 7B+

#### 2. Tools (the "hands")
External functions the agent can call:
- Web search
- Calculator
- Database query
- API calls
- File system
- Custom code execution

Example tool definition:
```json
{
  "name": "search",
  "description": "Search the web for information",
  "parameters": {
    "query": "string (search query)"
  }
}
```

#### 3. Memory (the "context")
Keeps track of:
- The original user query
- All past thoughts/actions/observations
- Current state

Example trajectory:
```
[User input: "What is the population of Tokyo?"]

Thought: I need to search for Tokyo's population
Action: search(query="Tokyo population 2024")
Observation: Tokyo's population is approximately 37.4 million

Thought: I have the answer
Final Answer: 37.4 million
```

#### 4. Executor (the "orchestrator")
Manages the loop:
- Parse LLM output (is it a thought? an action? a final answer?)
- Call the appropriate tool
- Pass observation back to LLM
- Loop until done

### Worked Example: Multi-Step Reasoning Agent

**Task:** "Is it a good time to visit Paris? Tell me the weather, any major events this week, and restaurant recommendations."

**Agent execution:**

```
Step 1:
Thought: I need to get the weather for Paris, upcoming events, and restaurant info.
Action: call [search("Paris weather this week")]
Observation: "Partly cloudy, 18–22°C, 60% chance of rain midweek"

Step 2:
Thought: Got the weather. Now I need events.
Action: call [search("Paris events this week July 2024")]
Observation: "Bastille Day celebration on July 14, outdoor concerts in parks"

Step 3:
Thought: Got weather and events. Now I need restaurant recommendations.
Action: call [search("best restaurants Paris 2024 reviews")]
Observation: "Top rated: L'Astrance (Michelin 3), Chez Janou (Mediterranean), Café de Flore (iconic)"

Step 4:
Thought: I have all the information I need to answer.
Final Answer: 
"This is a great time to visit Paris! 
Weather: Mild (18–22°C), bring a light jacket due to midweek rain.
Events: Bastille Day celebrations and outdoor concerts this week.
Restaurants: Try L'Astrance for fine dining, Chez Janou for Mediterranean, or Café de Flore for classic Paris experience."
```

### When to Use Agents

**Use agents when:**
- ✓ The task requires multiple steps / information gathering
- ✓ The LLM needs up-to-date info (web search, database queries)
- ✓ You need explainability (can see the reasoning chain)
- ✓ Complex reasoning: math, coding, data analysis (tool: code executor)

**Don't use agents when:**
- ✗ Simple generation task ("write me a poem")
- ✗ Real-time latency critical (<100ms needed) — agent loop adds latency
- ✗ Closed-domain task (LLM has all knowledge in weights, no search needed)

### Challenges with Agents

1. **Hallucination in tool selection**: LLM calls the wrong tool or makes up tools
   - **Fix:** Strictly validate tool names/params before calling

2. **Infinite loops**: Agent gets stuck in a loop (keeps taking same action)
   - **Fix:** Limit max iterations (e.g., 10 steps max)

3. **Error handling**: Tool fails (search returns no results, API timeout)
   - **Fix:** Graceful fallback ("No info found, try different query" or "Search failed, proceeding with what we know")

4. **Latency**: Each tool call adds network/compute latency
   - **Fix:** Parallel tool calls (if tools are independent) or async execution

### Worked Example: Agent + RAG for Customer Service

**Task:** "Where do I go to see my recent purchases and how do I request a refund?"

**Traditional chatbot:**
```
Chatbot: "Go to Account > Order History for purchases. Email support@company.com for refunds."
(Generic, may not match current UI/policy)
```

**Agent + RAG:**
```
Step 1:
Thought: User is asking about two things: viewing purchases and requesting refunds.
Action: retrieve_doc(query="where to view recent purchases")
Observation: "Recent purchases are in the mobile app: Home > Your Orders, or web: Account > Purchases (recently updated Sept 2024)"

Step 2:
Thought: Now I need refund policy.
Action: retrieve_doc(query="how to request refund")
Observation: "Refunds can be requested through the app (Order Details > Request Refund) or by contacting support@company.com. Window: 30 days from purchase. Processing: 5–7 business days."

Step 3:
Thought: I have both pieces of current, accurate information.
Final Answer: 
"To view purchases: 
- Mobile: Home > Your Orders
- Web: Account > Purchases

To request a refund (within 30 days):
- Mobile: Open the order > Request Refund
- Web: Go to Account > Purchases > Click order > Request Refund
- Or email support@company.com
Refunds process in 5–7 business days."
```

**Benefits:**
- ✓ Current (pulls latest docs)
- ✓ Explainable (you can see the docs it found)
- ✓ Flexible (works for new products/policies without retraining)

---

## 8. Interview Narrative: Tying It Together

### Quick Talking Points

**"What are foundational LLMs?"**
> Foundational LLMs like GPT and LLaMA are large transformer models pre-trained on diverse internet text to predict the next token. They develop general-purpose skills—reasoning, code, facts, style—from this simple objective at scale. Once trained, they can solve diverse downstream tasks via prompting or fine-tuning.

**"How do you process 'rhyming words for bat'?"**
> The model tokenizes the text, embeds tokens as vectors, adds positional information, runs through transformer attention layers (which learn which parts of the input matter), and generates the next token probabilistically. It repeats this loop: output token → input for next prediction → new token, until it generates a stopping signal. The attention mechanism is key—it figures out that "bat" and "rhyming" are the important tokens for this task.

**"How do you evaluate an LLM response?"**
> It depends on the task. For generation with references (translation, summarization), use BLEU or ROUGE. For open-ended tasks (QA, dialogue), human evaluation is gold standard, but LLM-as-judge is a fast, practical alternative. Perplexity is useful for model comparison but not task-specific quality.

**"What's a vector database, and why not use SQL?"**
> SQL databases are built for exact/range matches. Vector DBs are optimized for similarity search in high-dimensional space. Common use case: RAG. Embed your knowledge base into a vector DB. When a user asks a question, embed the question and retrieve the most similar documents, then pass them to an LLM for grounded generation. This reduces hallucination and keeps information current.

**"What's an agentic workflow?"**
> An agent is an LLM that can plan, act, and reflect. It decides what tools to call (search, calculator, API, database) based on the user's goal. It observes the results and uses them to decide the next action. Useful for multi-step reasoning, real-time info (web search), and explainability. Latency is a tradeoff—more powerful but slower than pure generation.

---

## 9. Summary Table: Quick Reference

| Concept | Key Insight | Use When | Interview Q |
|---------|-------------|----------|------------|
| **Foundational LLMs** | Pre-trained on broad data → general-purpose via prompting/fine-tuning | You want flexible, zero-shot-capable AI | "What makes them foundational?" |
| **Text processing** | Tokenize → embed → attention → generate (repeat) | Explaining model behavior | "Walk me through how LLMs process input." |
| **Evaluation** | Task-dependent: BLEU/ROUGE for generation, human/LLM-judge for open-ended | Comparing model quality | "How do you evaluate?" |
| **Vector DBs** | Similarity search in high-dim space; enables RAG | Grounded generation, semantic search | "Why not SQL?" |
| **Agentic workflows** | LLM + tools + reasoning loop | Complex tasks, real-time info, explainability | "When would you use agents?" |

---

## References & Resources

- **Vaswani et al. (2017)**: "Attention is All You Need" (Transformer paper)
- **Brown et al. (2020)**: "Language Models are Few-Shot Learners" (GPT-3 paper)
- **Touvron et al. (2023)**: "LLaMA: Open and Efficient Foundation Language Models"
- **Lewis et al. (2020)**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **Wei et al. (2022)**: "Emergent Abilities of Large Language Models" (scaling laws)

---

**Next:** Fine-tune this document based on your interview needs. Add more worked examples, adjust depth based on interviewer feedback, and practice verbal delivery of the narratives.
