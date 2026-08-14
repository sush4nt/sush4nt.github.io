# Agentic Systems & GenAI Engineering: Production Depth

**Date:** July 19, 2026
**Purpose:** Interview-ready reference for the "Agentic AI and GenAI" + "End-to-End MLOps" pillars of a Senior ML Scientist / Senior Data Scientist role. Complements `03_genai_foundations_v1.md` (LLM/transformer fundamentals) with the engineering layer: frameworks, orchestration, tool-calling, RAG, and production concerns.

**How to use this document:** Section 1–2 answer "what do you know about the frameworks and patterns." Section 3–4 answer "have you actually built and shipped this." Section 5 answers "how do you think about GenAI strategically" — the roadmapping angle the JD explicitly asks for. Sections are written surface → in-depth, same convention as the Transformers section in the foundations doc.

---

## Table of Contents

- [1. Framework Landscape: LangChain vs. CrewAI vs. AutoGen](#1-framework-landscape-langchain-vs-crewai-vs-autogen)
- [2. Multi-Agent Orchestration Patterns](#2-multi-agent-orchestration-patterns)
- [3. Tool-Calling Mechanics](#3-tool-calling-mechanics)
- [4. RAG Architecture Deep-Dive](#4-rag-architecture-deep-dive)
- [5. Fine-Tuning vs. RAG vs. Prompting: Decision Framework](#5-fine-tuning-vs-rag-vs-prompting-decision-framework)
- [6. Production Concerns for Agentic Systems](#6-production-concerns-for-agentic-systems)
- [7. Evaluating Agentic Systems](#7-evaluating-agentic-systems)
- [8. GenAI Use-Case Roadmapping (Strategy Layer)](#8-genai-use-case-roadmapping-strategy-layer)
- [9. Interview Narrative: Tying It Together](#9-interview-narrative-tying-it-together)
- [10. Summary Table: Quick Reference](#10-summary-table-quick-reference)

---

## 1. Framework Landscape: LangChain vs. CrewAI vs. AutoGen

### Why This Matters

The JD names all three explicitly. You don't need to have shipped all three, but you need a crisp, correct mental model of what each one is actually abstracting — because "I've used LangChain" without being able to say *why* it exists, or *when you'd reach for something else*, reads as surface-level tool familiarity rather than engineering judgment.

### The Core Distinction

All three frameworks solve the same underlying problem — orchestrating an LLM's reasoning loop plus tool calls plus (sometimes) other LLMs — but they make different default assumptions about **how much structure to impose**.

**Surface:**
- **LangChain** — a general-purpose toolkit: chains, agents, memory, retrievers, and (via LangGraph) explicit state-machine/graph orchestration. Lowest-level, most flexible, steepest learning curve.
- **CrewAI** — a role-based abstraction: you define agents as "crew members" with a role, goal, and backstory, and a process (sequential or hierarchical) for how they collaborate. Optimized for readability and fast prototyping of multi-agent workflows.
- **AutoGen** — a conversation-centric abstraction: agents are defined as conversable entities that pass messages to each other; orchestration emerges from a "group chat" pattern with a manager agent deciding who speaks next. Strong for research-style multi-agent experimentation and human-in-the-loop patterns.

**In-Depth:**

| Dimension | LangChain (+LangGraph) | CrewAI | AutoGen |
|---|---|---|---|
| **Core abstraction** | Chains/graphs of steps; explicit control flow | Roles + goals + process (sequential/hierarchical) | Conversable agents exchanging messages |
| **Orchestration model** | You define the graph/state machine explicitly | Framework infers execution order from `Process.sequential`/`Process.hierarchical` | A manager/group-chat agent dynamically picks the next speaker |
| **Best for** | Fine-grained control, production pipelines, complex conditional logic, integrating retrieval/tools/memory in a custom flow | Fast prototyping of role-based workflows (e.g., "researcher" → "writer" → "editor") | Research-style multi-agent reasoning, negotiation, human-in-the-loop debugging via chat transcripts |
| **Learning curve** | Steep — lots of abstractions (runnables, chains, graphs) | Shallow — very readable, declarative agent definitions | Moderate — conversation patterns are intuitive but debugging emergent behavior is harder |
| **Determinism** | High if you use LangGraph (explicit edges) | Medium (process type constrains order, but agent outputs are still stochastic) | Lower — next speaker/flow can be dynamically decided by the LLM itself |
| **Production maturity** | Most mature ecosystem, most integrations (vector DBs, tools, LLM providers) | Newer, lighter-weight, fewer integrations out of the box | Strong for prototyping, historically less common in hardened production pipelines |

**Interview-ready one-liner:** "LangChain/LangGraph gives you explicit control over the execution graph, which I'd reach for in production where I need deterministic, debuggable flows. CrewAI is faster to prototype role-based workflows where the collaboration pattern is naturally sequential or hierarchical. AutoGen shines when the reasoning itself benefits from an open-ended conversation between agents — useful for exploratory or research-style tasks, but the emergent, less-deterministic flow makes it harder to productionize without added guardrails."

**Likely follow-up: "Which would you use for X?"**
- Automating a fixed, auditable business process (e.g., document review → extraction → validation → routing) → **LangGraph** (explicit, deterministic, easy to log/debug each node)
- Simulating a team of specialized analysts producing a report → **CrewAI** (role/goal framing maps naturally)
- Open-ended research or brainstorming where agents should challenge each other → **AutoGen** (conversation-native)

---

## 2. Multi-Agent Orchestration Patterns

### Why Multi-Agent at All?

**Surface:** A single agent with many tools can get overloaded — too many instructions, too much context, and it starts making mistakes on tool selection or loses track of the overall goal. Splitting responsibilities across specialized agents (each with a narrower role, tool-set, and prompt) tends to produce more reliable results, at the cost of more orchestration complexity and latency.

**In-Depth — when single-agent-with-tools is enough vs. when you need multi-agent:**

| Signal | Single Agent + Tools | Multi-Agent |
|---|---|---|
| Task complexity | Few steps, tools don't conflict in purpose | Many steps, distinct phases (research → draft → critique) |
| Prompt/context load | Fits comfortably in one system prompt | Instructions for different roles would dilute each other |
| Need for specialization | General-purpose reasoning suffices | Distinct expertise needed per phase (e.g., SQL-writing agent vs. a business-summary agent) |
| Latency tolerance | Low — every extra agent hop adds latency | Higher — willing to trade latency for quality/reliability |
| Debuggability need | Moderate | High — want to isolate which "role" failed |

### Core Orchestration Patterns

**1. Sequential (Pipeline) Pattern**
```
Agent A (Researcher) → Agent B (Analyst) → Agent C (Writer) → Final Output
```
- Each agent's output becomes the next agent's input. Simple, deterministic, easy to debug (you can inspect the intermediate output at each stage).
- **Use when:** the task naturally decomposes into stages where each stage fully depends on the previous one's output.
- **Failure mode:** error propagation — if Agent A hallucinates a fact, Agent B and C build on a wrong foundation. Mitigation: validation/critique step between stages.

**2. Hierarchical (Manager/Worker) Pattern**
```
        Manager Agent
       /      |       \
Worker A   Worker B   Worker C
```
- A manager agent decomposes the task, delegates sub-tasks to worker agents, and synthesizes their outputs.
- **Use when:** sub-tasks are independent and can be parallelized, or when the decomposition itself requires reasoning (the manager decides *what* sub-tasks are needed, not just running a fixed pipeline).
- **This is the pattern most naturally suited to CrewAI's `Process.hierarchical`.**
- **Failure mode:** manager makes a poor decomposition decision — garbage-in from the top propagates to all workers. Mitigation: give the manager a fixed menu of valid decompositions rather than fully open-ended planning, if the domain allows it.

**3. Debate / Critique Pattern**
```
Agent A (Proposer) ⇄ Agent B (Critic) → iterate → Final Answer
```
- Two (or more) agents argue or critique each other's outputs before converging on a final answer. Often used to reduce hallucination or catch reasoning errors — one agent explicitly tasked with poking holes in the other's output.
- **Use when:** correctness/quality matters more than latency, and a single pass is prone to errors (e.g., financial analysis, code review, complex reasoning).
- **Failure mode:** infinite disagreement loop, or two weak agents converging on a confidently wrong answer together ("groupthink"). Mitigation: cap iterations, and consider using a stronger/different model as the critic than the proposer.

**4. Group Chat / Emergent Pattern (AutoGen-native)**
```
Manager picks next speaker dynamically from {Agent A, Agent B, Agent C, ...}
```
- No fixed order — a manager (often itself an LLM call) decides who should speak next based on conversation state.
- **Use when:** the right next step genuinely depends on what's been said so far and can't be pre-determined (open-ended problem solving).
- **Failure mode:** least deterministic pattern — hardest to test/debug/guarantee behavior in production, best reserved for internal/exploratory tools rather than customer-facing systems.

**Interview-ready one-liner:** "I pick the orchestration pattern based on how deterministic I need the flow to be. Sequential for clean pipelines, hierarchical when sub-tasks are independent and need dynamic decomposition, debate/critique when correctness matters more than speed, and group-chat/emergent only for exploratory internal tooling where I can tolerate non-determinism."

---

## 3. Tool-Calling Mechanics

### Why This Is Interview Gold

This is the most concrete, "have you actually built this" question in the agentic space. Vague answers ("the LLM calls a function") get probed immediately on error handling and schema design — the parts that separate a demo from a production system.

### The Mechanics

**Surface:** You describe each tool to the LLM as a structured schema (name, description, parameter types). The LLM doesn't execute code — it outputs a *structured request* to call a tool with specific arguments. Your orchestration layer parses that request, actually executes the function, and feeds the result back into the LLM's context as an "observation."

**In-Depth — the full loop:**

1. **Schema definition** — each tool is described in a structured format (JSON schema is the near-universal standard):
```json
{
  "name": "get_campaign_performance",
  "description": "Retrieve performance metrics for an ad campaign by ID and date range",
  "parameters": {
    "type": "object",
    "properties": {
      "campaign_id": {"type": "string"},
      "start_date": {"type": "string", "format": "date"},
      "end_date": {"type": "string", "format": "date"}
    },
    "required": ["campaign_id", "start_date", "end_date"]
  }
}
```
- **Why description quality matters more than people expect:** the LLM decides *whether* and *how* to call a tool almost entirely based on the natural-language description. Vague descriptions → wrong tool selection or malformed arguments. This is a real production lesson worth stating explicitly if asked.

2. **Structured output enforcement** — modern LLM APIs support constrained/structured generation (function-calling modes, JSON mode, grammar-constrained decoding) so the model's output is guaranteed to be parseable JSON matching your schema, rather than hoping the model formats it correctly in free text.

3. **Execution & validation** — your code receives the tool call request, **validates arguments before executing** (never trust LLM-generated arguments blindly — e.g., check `campaign_id` exists, date range is sane), executes the actual function/API call, and captures the result (or the error).

4. **Feeding results back** — the tool's output (or error message) is appended to the conversation as an "observation," and the LLM continues reasoning with that new information in context.

### Error Handling — Where Production Systems Actually Differ From Demos

| Failure Mode | What Happens | Production Mitigation |
|---|---|---|
| LLM calls a tool that doesn't exist (hallucinated tool name) | Orchestrator can't find a matching function | Validate tool name against registered tools before execution; return a clear error observation ("tool X does not exist, available tools are: ...") so the model can self-correct |
| LLM provides malformed/wrong-type arguments | Schema validation fails | Reject and return the validation error as an observation rather than crashing — let the model retry with corrected arguments |
| Tool call succeeds but returns an unexpected/empty result | Model may hallucinate a plausible-sounding answer instead of acknowledging the gap | Explicitly instruct the model (in system prompt) on how to handle empty results; test this path directly |
| Tool times out / API is down | Silent failure or hang | Set explicit timeouts, return an error observation, and design a fallback (retry, alternate tool, or graceful "I don't have this information right now") |
| Model gets stuck calling the same tool repeatedly | Infinite loop, runaway cost | Cap max iterations/tool calls per request; detect repeated identical calls and break the loop |

**Interview-ready one-liner:** "The hard part of tool-calling isn't wiring up the function — it's building the guardrails around it: validating arguments before execution, capping iterations to avoid runaway loops, and making sure error states are surfaced back to the model as clear observations rather than crashing the pipeline. Tool description quality also matters a lot — that's genuinely where most tool-selection errors come from in my experience."

---

## 4. RAG Architecture Deep-Dive

*(This extends Section 6 of `03_genai_foundations_v1.md`, which covers the "what and why" of vector DBs. This section goes deeper into the engineering choices that determine whether a RAG system actually works well.)*

### The Real RAG Pipeline (Beyond "Embed and Retrieve")

**Surface:** RAG has three stages that each have significant design choices: (1) chunking your documents, (2) retrieval, and (3) generation using the retrieved context. Most RAG quality problems trace back to chunking or retrieval, not the LLM itself.

**In-Depth — Chunking Strategies:**

| Strategy | How It Works | Trade-off |
|---|---|---|
| **Fixed-size chunking** | Split text every N tokens/characters, often with overlap | Simple, fast, but can split a sentence or idea mid-way, hurting retrieval relevance |
| **Recursive/semantic chunking** | Split along natural boundaries (paragraphs, sections) first, falling back to smaller units only if a chunk is still too large | Preserves semantic coherence better; more implementation complexity |
| **Sentence-window chunking** | Retrieve based on individual sentence embeddings, but include surrounding sentences as context when passed to the LLM | Improves retrieval precision (matching on a specific sentence) while preserving enough context for generation |
| **Document/hierarchical chunking** | Chunk at multiple granularities (e.g., section summary + detailed chunks) and retrieve at the level appropriate to the query | Handles both "what is X" (needs detail) and "summarize this doc" (needs the summary level) well; more indexing complexity |

**Why chunk size is a real trade-off, not just an implementation detail:** Too small → chunks lack enough context to be individually meaningful, and you retrieve fragments. Too large → you dilute the embedding (averaging semantics over too much text) and waste context window on irrelevant surrounding text, and increase cost.

**In-Depth — Retrieval Quality:**

- **Pure vector similarity (dense retrieval)** can miss exact-match cases — e.g., a query containing a specific product SKU or code might not be the *semantically* closest chunk by embedding, even though it's the obviously correct one.
- **Hybrid search** combines dense (vector) retrieval with sparse/keyword retrieval (e.g., BM25) and merges the results — this catches both semantic matches and exact-term matches. A common production pattern: retrieve top-K from each, then combine/re-rank.
- **Re-ranking:** retrieve a larger candidate set cheaply (e.g., top-50 via vector similarity), then use a more expensive but more accurate model (a cross-encoder, or an LLM call) to re-rank and select the true top-K (e.g., top-5) to actually pass to the generator. This two-stage "retrieve cheap, rank expensive" pattern is standard in production search and directly transfers to RAG.
- **Query transformation:** the raw user query is sometimes a poor retrieval query (too short, ambiguous, or conversational). Techniques like query rewriting/expansion (using an LLM to reformulate the query before embedding it) or generating multiple sub-queries for a complex question can meaningfully improve retrieval.

**Interview-ready one-liner:** "Most RAG failures I've seen aren't the LLM's fault — they're retrieval failures. My default approach is hybrid retrieval (dense + keyword) with a re-ranking stage, and chunking that respects document structure rather than fixed-size splits. If the query is ambiguous or the question requires synthesizing multiple facts, I'd add query rewriting or decomposition before retrieval."

---

## 5. Fine-Tuning vs. RAG vs. Prompting: Decision Framework

The JD specifically requires "fine-tuning models for specific domain tasks" — so you need a clear, defensible framework for *when fine-tuning is actually the right call* versus RAG or prompting, since fine-tuning is the most expensive/slowest of the three to iterate on.

### Decision Framework

```
What kind of gap are you trying to close?

├─ Model lacks KNOWLEDGE (facts, docs, current data)?
│  └─ → RAG. Fine-tuning is a poor fit for injecting facts — it's expensive,
│       doesn't reliably "add" specific facts, and goes stale immediately.
│       RAG keeps knowledge current and is auditable (you can show sources).
│
├─ Model lacks the right BEHAVIOR/FORMAT/STYLE consistently?
│  (e.g., always respond in a specific JSON schema, a specific tone,
│   a specific domain "voice")
│  └─ → Prompting/few-shot first. If that's not reliable enough at scale
│       → Fine-tuning (this is the classic fine-tuning use case:
│       teaching behavior, not facts)
│
├─ Model needs domain-specific REASONING patterns not well represented
│  in its pretraining data (e.g., specialized clinical reasoning,
│  a niche technical domain with unusual terminology/structure)?
│  └─ → Fine-tuning (ideally combined with RAG for the factual layer)
│
├─ Latency/cost constraints require a much SMALLER model to hit
│  quality bar that only a larger model achieves out of the box?
│  └─ → Fine-tuning a smaller model (distillation-style: use a large
│       model's outputs as training data for a small model)
│
└─ Just need it to work for a handful of well-specified tasks, fast?
   └─ → Prompting/few-shot. Cheapest and fastest to iterate — always
        the right starting point before reaching for RAG or fine-tuning.
```

### Why This Framework Is Interview-Strong

It directly rebuts a common naive answer ("fine-tune it on our data to make it know about X"), which is usually wrong — fine-tuning is notoriously unreliable at reliably injecting new facts (the model may "learn" a fact inconsistently, or forget old knowledge — catastrophic forgetting), whereas RAG guarantees the facts are literally in the context at generation time.

**Interview-ready one-liner:** "I treat prompting as the default and cheapest lever, RAG as the fix for knowledge/currency gaps, and fine-tuning as the fix for behavior, format, or domain-reasoning gaps that prompting can't reliably achieve at scale. A common mistake is reaching for fine-tuning to inject facts — that's what RAG is actually good at; fine-tuning on facts is expensive and unreliable, and the knowledge goes stale the moment you ship."

### Fine-Tuning Approaches (If Pushed Further)

| Approach | What It Does | When to Use |
|---|---|---|
| **Full fine-tuning** | Update all model weights | Rare in practice now — expensive, needs a lot of data, risk of catastrophic forgetting |
| **LoRA (Low-Rank Adaptation)** | Freeze base weights, train small low-rank adapter matrices injected into attention/FFN layers | The default modern approach — much cheaper, faster, and multiple LoRA adapters can be swapped for different tasks on the same base model |
| **Instruction fine-tuning (SFT)** | Fine-tune on (instruction, ideal response) pairs | Teaching the model to follow a specific format/behavior consistently |
| **RLHF / DPO (preference-based)** | Fine-tune using human (or AI) preference comparisons rather than single "correct" answers | Aligning tone/quality/safety along a preference dimension rather than a single ground truth |

---

## 6. Production Concerns for Agentic Systems

*This is where your existing production/MLOps background (KServe, MLflow, Prometheus, Grafana, latency guardrails) directly transfers — the underlying discipline is identical, just applied to a non-deterministic, multi-step system instead of a single model call.*

### Cost Control

**Surface:** Agentic systems can call the LLM many times per user request (each reasoning step, each tool call round-trip). Cost scales with number of steps × tokens per step, and a poorly-bounded agent can spiral into very expensive requests.

**In-Depth:**
- **Token accounting per agent loop:** track input + output tokens per step, not just per request — this is what actually reveals which stage of a multi-agent pipeline is expensive (often the "context accumulation" problem: each step re-sends growing conversation history, so token cost grows superlinearly with steps).
- **Mitigations:** cap max iterations/tool calls; summarize/compress conversation history periodically instead of keeping full history; use a cheaper/smaller model for simple sub-tasks (e.g., a routing decision) and reserve the expensive model for the step that actually needs strong reasoning — this "model cascading" or "model routing" pattern is a strong production talking point.

### Observability & Tracing

**Surface:** Unlike a single model call, an agentic system's failure could be in the plan, the tool call, the retrieval, or the final synthesis — you need visibility into *each step*, not just the final output, to debug anything.

**In-Depth:**
- Equivalent of your Prometheus/Grafana monitoring, but for reasoning traces: log every thought/action/observation step, with latency and token cost per step (tools like LangSmith, or a custom structured-logging approach serve this purpose).
- **What to actually monitor in production** (directly analogous to your existing latency/error monitoring):
  - Step-level latency (which stage is the bottleneck)
  - Tool-call success/failure rate (is a specific tool unreliable or is the model calling it wrong?)
  - Loop/iteration count distribution (is the agent frequently hitting the max-iteration cap — a sign of task ambiguity or a missing tool)
  - Final-output quality proxy (e.g., did the agent reach a "final answer" state, or did it time out/fail)

### Guardrails Against Runaway Behavior

- **Max iteration/tool-call caps** (mentioned above) — a hard ceiling to prevent infinite loops.
- **Action allow-lists** — especially for agents with access to consequential tools (sending emails, modifying records), restrict what actions are possible, and consider requiring human confirmation for irreversible actions — same principle as the "explicit permission required" tier in your own production risk thinking.
- **Timeouts at every external call** — a hung tool call shouldn't hang the whole agent loop.
- **Sandboxing for code-execution tools** — if an agent can execute code (e.g., for data analysis tasks), that execution needs to be sandboxed/isolated from production systems.

**Interview-ready one-liner:** "I treat an agent pipeline the same way I'd treat any production ML system in terms of rigor — I just add step-level observability because failures can happen at the plan, the tool call, or the synthesis stage. Cost and latency guardrails (max iterations, model routing to cheaper models for simple steps) are non-negotiable, because agentic loops can silently multiply cost in a way a single model call never does."

---

## 7. Evaluating Agentic Systems

### Why This Is Harder Than Single-Turn Eval

**Surface:** A single LLM response can be evaluated against a reference or by an LLM-judge relatively directly (see Section 5 of the foundations doc). An agentic system's *trajectory* — the sequence of decisions it made — matters as much as the final answer. Two agents can reach the same correct final answer via very different quality of reasoning (one efficient and grounded, one lucky and wasteful).

**In-Depth — What to Evaluate:**

| Dimension | What It Measures | How to Measure |
|---|---|---|
| **Task success rate** | Did the agent achieve the actual goal? | Binary/graded success against a labeled test set of tasks with known correct outcomes |
| **Trajectory efficiency** | Did it take a reasonable number of steps, or wander? | Compare step count to a reasonable baseline; flag excessive tool calls |
| **Tool-selection accuracy** | Did it call the right tool, with correct arguments, at each step? | Requires step-level annotation/tracing (ties back to Section 6's observability) |
| **Groundedness** | Are claims in the final answer actually supported by retrieved/observed data, or hallucinated? | LLM-as-judge comparing final answer against the actual observations in the trajectory |
| **Robustness** | Does it recover gracefully from a tool failure or unexpected observation? | Adversarial test cases: inject a tool failure or ambiguous result and check recovery behavior |

**Interview-ready one-liner:** "Evaluating an agent isn't just 'was the final answer right' — I look at the trajectory: did it use the right tools efficiently, is the final answer actually grounded in what it observed (not hallucinated on top of real retrieval), and does it degrade gracefully when a tool fails. That last one — robustness to failure — is the one most teams skip in eval and then get burned by in production."

---

## 8. GenAI Use-Case Roadmapping (Strategy Layer)

*The JD explicitly wants you to "identify and implement high-impact GenAI use cases" — this is a strategic/product-judgment question, not a technical one, and senior candidates are expected to have a framework for it.*

### A Simple Prioritization Framework

When asked "how would you identify GenAI opportunities in [X]," structure your answer around three axes:

1. **Feasibility** — Is this a task where an LLM's core strength (language understanding/generation, pattern synthesis across unstructured data) is actually the bottleneck? Avoid use cases that are really structured-data/deterministic-logic problems wearing a GenAI costume.
2. **Value concentration** — Is this a high-frequency, currently-manual, unstructured-data-heavy workflow? (GenAI's clearest ROI is almost always at the intersection of "lots of unstructured text/data" + "currently done manually by skilled humans" + "tolerance for occasional imperfection.")
3. **Risk tolerance of the surface** — Internal tooling (e.g., an internal agent that drafts a report for a human to review) tolerates more error than a customer-facing, irreversible-action system. Start high-value/low-risk, expand to higher-risk surfaces once reliability is proven.

**A concrete worked example, framed for an AdTech/measurement context (relevant to this JD):**
- **Candidate use case:** "Auto-generate campaign performance narratives for account managers" (unstructured synthesis over structured data + text) → high feasibility (LLM strength = synthesis/narrative), high value concentration (currently manual, done by every account manager, high frequency), moderate risk (internal, human-reviewed before client-facing) → **strong candidate.**
- **Candidate use case (a trap to name explicitly if asked):** "Use an LLM to calculate optimal bid prices in real-time bidding" → low feasibility (this is a numerical optimization problem, not a language problem — XGBoost/statistical models are the right tool, not an LLM) → **weak candidate, good to explicitly reject and explain why**, since this shows judgment rather than GenAI-maximalism.

**Interview-ready one-liner:** "I look for the intersection of high-frequency, currently-manual, unstructured-data-heavy workflows — that's where LLMs have genuine leverage. I'm equally comfortable saying where GenAI is the *wrong* tool — e.g., a real-time bid-pricing decision is a numerical optimization problem, not a language problem, and forcing an LLM into that is a common anti-pattern I'd push back on."

---

## 9. Interview Narrative: Tying It Together

**"Walk me through how you'd design a multi-agent system for [some business process]."**
> I'd start by asking whether this genuinely needs multiple agents or if a single agent with well-scoped tools is enough — multi-agent adds real orchestration and latency cost. If the task decomposes into distinct phases needing different expertise, I'd pick an orchestration pattern based on how deterministic I need it: sequential for a clean pipeline, hierarchical if sub-tasks are independent, or debate/critique if correctness matters more than speed. I'd build in step-level observability from day one, cap iterations to control cost and prevent runaway loops, and evaluate the system on trajectory quality — not just final-answer correctness.

**"When would you fine-tune vs. use RAG?"**
> RAG for knowledge and currency — anything that needs facts or data that changes over time. Fine-tuning for behavior, format, or domain-specific reasoning patterns that prompting can't reliably achieve at scale. Fine-tuning is a poor tool for injecting facts specifically — it's expensive to iterate on and the knowledge goes stale immediately, whereas RAG keeps things current and auditable.

**"How do you control cost in an agentic pipeline?"**
> Token accounting per step, not just per request, since context accumulates across steps. Model routing — cheap models for routing/simple decisions, expensive models reserved for the step that actually needs strong reasoning. Hard caps on iterations and tool calls to prevent runaway loops, which is the single most common way agentic costs spiral silently.

**"How is evaluating an agent different from evaluating a single LLM call?"**
> A single call, you evaluate the output. An agent, you evaluate the trajectory — tool-selection accuracy, step efficiency, whether the final answer is actually grounded in what it observed rather than hallucinated on top of real retrieval, and critically, how it degrades when a tool fails. Robustness to failure is the dimension most teams skip and then get burned by in production.

---

## 10. Summary Table: Quick Reference

| Concept | Key Insight | Interview Trigger |
|---|---|---|
| **LangChain/CrewAI/AutoGen** | Same problem, different structure defaults: explicit graph vs. role-based vs. conversation-native | "Which framework would you use for X?" |
| **Orchestration patterns** | Sequential (pipeline), hierarchical (manager/worker), debate/critique, group-chat — pick based on determinism/latency/quality trade-off | "Design a multi-agent system for X" |
| **Tool-calling** | Schema quality drives tool-selection accuracy; validate args before executing; cap iterations | "How do you handle tool-calling errors?" |
| **RAG engineering** | Most RAG failures are retrieval failures, not LLM failures — chunking strategy and hybrid search + re-ranking matter most | "How would you improve a RAG system's accuracy?" |
| **Fine-tune vs. RAG vs. prompt** | RAG = knowledge/currency, fine-tune = behavior/format/domain-reasoning, prompt = default starting point | "When would you fine-tune?" |
| **Production guardrails** | Step-level observability, cost/token accounting per step, iteration caps, action allow-lists for consequential tools | "How do you productionize an agent?" |
| **Agent evaluation** | Trajectory quality (tool accuracy, groundedness, robustness to tool failure), not just final-answer correctness | "How do you evaluate an agentic system?" |
| **GenAI roadmapping** | High-frequency + manual + unstructured-data-heavy = high leverage; explicitly reject GenAI for numerical-optimization problems | "How would you identify GenAI use cases?" |

---

**Next:** Pair this with your Adform production experience — for every "production concerns" talking point above, have a 30–60 second version anchored to a KServe/MLflow/Prometheus example, even if the underlying system wasn't agentic. The discipline (observability, guardrails, cost control) is what's being assessed, and you already have real stories for that discipline.
