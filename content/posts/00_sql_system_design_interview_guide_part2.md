# Senior SQL: System Design & Problem-Solving Interview Guide

**Purpose:** Reference document for navigating complex, production-grade SQL architectural decisions. Based on real Amazon assessment scenarios and SQL system design principles.

---

## Core Decision Framework

Before diving into specific scenarios, understand the meta-framework:

1. **Identify the bottleneck** — What's the actual constraint? (query logic, data volume, schema design, permissions)
2. **Match the tool to the problem** — The "correct" answer solves the named problem, not a different one
3. **Favor set-based over procedural** — SQL excels at bulk operations; loops/cursors are red flags
4. **Non-disruptive first** — Audit/measure before making irreversible changes
5. **Audience matters** — Documentation, naming, and design choices depend on who consumes the output

---

## Scenario 1: System Monitoring Database Performance

**Problem:** Application logs table holds 900GB of data (18 months). Real-time monitoring needs fast queries, but compliance requires keeping 3 months hot, 12 months accessible, remainder archived.

**Key Tension:** Performance vs. compliance retention.

### Answer: Table Partitioning by Month + Archive to Separate Tablespaces

**Why This Wins:**
- Queries on recent data (the peak-hour bottleneck) scan only relevant partitions, not the full 900GB
- Directly maps to the tiered retention policy: 3-month hot partitions on fast storage, 12-month partitions on standard, older data moved to archive tablespaces
- Non-destructive; data never deleted, compliance satisfied
- Reversible and doesn't change the table's logical structure

**Why Others Fail:**
- **Compression alone:** Reduces size but doesn't prevent full-table scans; engine still touches old data
- **Moving to historical_logs table on same server:** Doesn't solve resource contention; archive still on same hardware
- **Deleting data >15 months:** Violates compliance; data is destroyed, not archived

**Interview Talking Points:**
- Explain the 3-tier access pattern: hot/warm/cold data placement
- Discuss partition key selection (by date for time-series data)
- Mention automated partition lifecycle management (create new, archive old)
- Note: works at scale; Twitter/Facebook use similar strategies for massive time-series

---

## Scenario 2: Financial Reporting Query Timeout

**Problem:** Multi-table join (customer, transaction, accounting) with ~15M rows times out after 10 minutes. Correlated subqueries used; adequate CPU/memory available.

**Key Tension:** Logic efficiency vs. brute-force resource allocation.

### Answer: Convert Correlated Subqueries to Equivalent Joins

**Why This Wins:**
- Correlated subqueries execute once *per outer row* = 15M+ executions; joins execute once
- CPU/memory being adequate signals the problem is *query logic*, not resources
- Set-based execution (joins) is SQL's strength; leverages optimizer and indexes
- No schema or output changes needed

**Why Others Fail:**
- **Forcing join order with hints:** Band-aid; doesn't fix the re-execution pattern
- **Adding indexes:** Helps, but if subqueries still execute per row, indexes won't save you
- **Rewriting with CTEs:** Improves readability but doesn't eliminate correlation (unless you also convert to joins)

**Interview Talking Points:**
- Explain subquery execution model: "Every outer row triggers N re-evaluations"
- Discuss optimizer limitations: harder to optimize across correlated scopes
- Walk through an example: `WHERE EXISTS (SELECT 1 FROM T2 WHERE T2.x = T1.x)` → `LEFT JOIN + WHERE T2.id IS NOT NULL`
- Mention: This is *the* common performance antipattern in financial/reporting systems

---

## Scenario 3: Order Processing Bottleneck Analysis

**Problem:** Analyze order processing steps to find which steps consistently exceed SLA. Table has 15M orders; need to track time between sequential status changes.

**Key Tension:** Temporal/sequential logic (hard in SQL) vs. efficiency.

### Answer: Window Functions to Calculate Step Duration

**Why This Wins:**
- **Window functions are *designed* for this:** `LEAD(timestamp) OVER (PARTITION BY order_id ORDER BY timestamp) - timestamp` gets the gap to the next row in one pass
- Partitioning by order_id + ordering by timestamp preserves sequence naturally
- One pass through data; no self-joins or recursive logic
- Joins to processing_steps on result set to compare against targets; clean separation

**Why Others Fail:**
- **Join + GROUP BY status:** Groups collapse the sequence; you lose step order
- **Self-join on order_status_history:** Technically works but clunky; requires matching "next" row via subquery/MIN; slower
- **Stored procedure loop:** Row-by-row procedural logic; doesn't scale, blocks on large datasets

**Interview Talking Points:**
- Explain window function frame: "Partition defines the group; ORDER BY defines sequence within partition"
- Show the syntax: `LEAD(col) OVER (PARTITION BY ... ORDER BY ...)`
- Contrast with correlated subqueries: "Window functions pre-compute rankings; subqueries re-compute per row"
- Mention: Window functions are increasingly standard across DB engines (PostgreSQL, SQL Server, MySQL 8+)

---

## Scenario 4: Product Catalog Integrity

**Problem:** 10,000 products × 50 categories. Each product must belong to exactly one category. Inventory team updates data frequently. Need to prevent invalid category assignments.

**Key Tension:** Data integrity at scale vs. application-layer validation.

### Answer: Foreign Key Constraint (category_id → categories table)

**Why This Wins:**
- **Database-level enforcement:** No invalid state possible, regardless of app logic or user action
- Self-maintaining: New categories added to categories table; FK automatically recognizes them
- Enforced on every write; can't bypass with direct SQL or privilege escalation
- Industry standard for relational integrity; auditors/compliance understand it

**Why Others Fail:**
- **Stored procedure for updates:** Depends on discipline; direct table access bypasses it; not a real constraint
- **Check constraint against hardcoded list:** Brittle; requires DDL changes when categories change; doesn't scale
- **Unique index on (product_id, category_id):** Enforces uniqueness, not validity; `category_id = 9999` is still allowed if it doesn't exist

**Interview Talking Points:**
- Explain FK semantics: "Guarantees referential integrity; prevents orphans"
- Discuss cascading deletes/updates: When to use (and cautions)
- Mention: FKs have performance overhead (index lookups on every insert); acceptable at 10K scale, worth discussing at 100M scale
- Note the distinction: Uniqueness ≠ validity

---

## Scenario 5: Database Role Permission Audit

**Problem:** Roles have excessive CREATE, ALTER, DELETE on production schemas. Need to fix without breaking operations. What comes first?

**Key Tension:** Security risk vs. avoiding disruption.

### Answer: Run Permissions Audit Query First

**Why This Wins:**
- **Non-disruptive fact-gathering:** Read-only; zero risk to production
- Maps current state before any action; prevents blind privilege revocation
- Identifies which grants are actually used vs. legacy cruft
- Informs all downstream decisions (which roles to create, what to audit, hierarchy design)
- Shows what depends on over-privileged access before you cut it

**Why Others Fail:**
- **Creating new minimal roles:** Premature; you don't know usage patterns yet; risk breaking workflows
- **Enabling extended logging:** Monitoring measure, not remediation; leaves current risk exposed
- **Setting up role hierarchies:** Architectural change; without audit data, over- or under-provisioning

**Interview Talking Points:**
- Emphasize: "Audit before action"
- Discuss what the audit should capture: Role name → schema → object → privilege → grantee chain
- Mention: Look for roles like `DBA_TEMP`, `DEV_LEGACY`, roles with unusually broad grants
- Talk through: "Once you have the audit, you can risk-rank grants: which are truly unused? Which operations depend on them?"

---

## Scenario 6: Inventory Report Column Naming

**Problem:** Query for daily inventory reports joins product/category tables; calculates metrics with complex expressions. Dashboard teams consume the output. Style guide is silent on alias naming.

**Key Tension:** Technical precision vs. business clarity.

### Answer: Descriptive Business Terms Matching Dashboard Terminology

**Why This Wins:**
- **End users are business teams viewing dashboards**, not SQL engineers
- Aliases matching dashboard labels means no translation layer; what they see = what's in the query output
- Reduces downstream errors: No "Is this `inv_qty` or `stock_units`?" confusion
- Self-documenting integration: Query outputs map directly to report fields

**Why Others Fail:**
- **Abbreviated names + table prefixes (inv_qty, prod_id):** Convenient for engineers but forces non-technical users to decode conventions
- **Standardized calculation prefixes (avg_daily_units):** Useful internally but assumes technical audience parsing prefixes
- **Technical names = source columns (quantity, category_code):** Exposes internal schema to business users; tight coupling; breaks when schema changes

**Interview Talking Points:**
- Discuss audience-driven design: "Your aliases should speak the language of your consumer"
- Give example: If dashboard shows "Units in Stock", alias should be `Units_in_Stock`, not `inv_qty`
- Mention: This applies to all downstream outputs—reports, dashboards, data feeds to other teams
- Note: Not about being fancy; it's about reducing translation overhead

---

## Scenario 7: Complex Query Documentation for Governance

**Problem:** 200-line query with 8 joins, multiple window functions, frequently updated based on changing reporting requirements. Needs governance documentation.

**Key Tension:** Technical details vs. business maintainability.

### Answer: Business Context, Section Descriptions, Parameter Explanations

**Why This Wins:**
- **Governance is about organization-wide understanding**, not just performance tuning
- Future maintainers (possibly not you) need to know *why* each section exists, what business problem it solves
- Frequent updates require someone to safely modify logic without breaking downstream quarterly insights
- Structural clarity (section-by-section breakdown) makes a 200-line query navigable

**Why Others Fail:**
- **Execution plan + DB version details:** Performance-focused; goes stale; doesn't help someone understand business intent
- **Optimization history + index recommendations:** Valuable for tuning but not governance/maintainability
- **Author contact + modification timestamps:** Metadata, not substance; doesn't explain what the query does

**Interview Talking Points:**
- Emphasize: "Documentation is for future humans, not machines"
- Structure recommendations: 
  - Header: Purpose (e.g., "Quarterly revenue forecast by region and product line")
  - Per section: "This section calculates X because Y. Input: {param_name}, output: {column_name}"
  - Parameters: "Discount_threshold (default 0.15): Marks items on promotional pricing; change affects year-over-year comparisons"
- Mention: This is increasingly enforced in regulated industries (finance, healthcare)

---

## Scenario 8: Handling Missing Data in Analysis

**Problem:** 50K customer satisfaction survey responses with 30% NULLs across 5 rating dimensions. Need averages, distributions, correlations per dimension. Team needs confidence in the numbers.

**Key Tension:** Statistical honesty vs. completeness illusion.

### Answer: Use AVG() and COUNT() — Document Actual Response Counts

**Why This Wins:**
- **SQL AVG() and COUNT() ignore NULLs correctly by default** — no fabrication
- Transparency is critical with 30% missing: COUNT(dimension) shows how many responses each statistic is based on
- Product team can see: "ease_of_use avg = 4.2 from 43K responses; customer_service avg = 3.8 from 31K responses" → understands variance in reliability
- Preserves true distribution; correlations between dimensions remain statistically valid

**Why Others Fail:**
- **CASE-based weighted averages:** Adds complexity and arbitrary weighting schemes not justified by problem
- **Separate segments (complete vs. partial):** Fragments analysis; loses statistical power; confounds correlation interpretation
- **Replacing NULLs with median imputation:** Distorts true distribution; artificially inflates/deflates correlations between dimensions (a core analysis goal here)

**Interview Talking Points:**
- Explain: "AVG() and COUNT() handle NULLs differently: AVG ignores them in numerator AND denominator; COUNT can count non-null rows"
- Example: `SELECT AVG(rating), COUNT(rating) FROM survey` → AVG is based on responses that answered; COUNT shows how many
- Warn against: Imputation without transparency; silently "fixing" missing data introduces bias, especially in correlation analysis
- Mention: In real BI/analytics contexts, you'd document confidence intervals around averages based on response count

---

## Quick Reference: Decision Tree for Senior SQL Questions

```
START: Complex SQL/System Design Problem
│
├─ Is the problem about PERFORMANCE?
│  ├─ Query running slow? → Check query LOGIC first (subqueries, joins order) 
│  │  │                   → Then check INDEX coverage
│  │  │                   → Only then check resource allocation
│  │
│  └─ Data volume issue? → Consider PARTITIONING (by date, by range)
│                        → Archive old data to separate tablespace
│
├─ Is the problem about DATA INTEGRITY?
│  ├─ Preventing invalid states? → Use CONSTRAINTS (FK, CHECK, UNIQUE)
│  │                             → NOT application-layer validation
│  │
│  └─ Tracking sequences/order? → Use WINDOW FUNCTIONS (LEAD/LAG/ROW_NUMBER)
│                               → NOT procedural loops
│
├─ Is the problem about SECURITY/GOVERNANCE?
│  ├─ Too many privileges? → AUDIT first (non-disruptive)
│  │                      → Then design minimal-privilege roles
│  │
│  └─ Complex query needs maintenance? → Document BUSINESS CONTEXT, not just SQL
│                                      → Explain each section's purpose
│
├─ Is the problem about DATA QUALITY/ANALYSIS?
│  ├─ Missing data (NULLs)? → Use set operations that handle NULLs correctly
│  │                       → Document the count of valid responses per metric
│  │                       → Avoid silent imputation
│  │
│  └─ Output consumed by non-technical users? → Use BUSINESS TERMINOLOGY in aliases
│                                            → Match dashboard language
│
└─ END: Your answer should match the *actual* problem, not a different one
```

---

## Patterns to Recognize (and the Answers They Point To)

| Pattern | Red Flag | Right Direction |
|---------|----------|-----------------|
| "Timing out after 10 min, CPU OK" | Inefficient query logic | Convert correlated subqueries → joins |
| "Need to track state changes over time" | Procedural thinking (loops) | Window functions (LEAD/LAG) |
| "Need to enforce a rule consistently" | App-layer validation | Database constraints (FK, CHECK) |
| "Huge table, queries slow, need to fix" | Immediate indexing impulse | First: partition by date/range; then index |
| "30% missing data, need averages" | Impulse to fill NULLs | Use AVG/COUNT correctly; document actual counts |
| "Complex query, team updates it frequently" | Performance/execution notes | Business context & section descriptions |
| "Too many user privileges" | Immediate revocation impulse | Audit first, then re-architect roles |
| "Query output consumed by dashboards" | Technical naming | Business terminology matching dashboard labels |

---

## Senior-Level Interview Talking Points (Meta)

When presenting any answer:

1. **State the constraint first:** "The problem is that [performance/integrity/compliance/clarity], not [something else]"
2. **Explain the mechanism:** Walk through *why* your approach solves that constraint
3. **Acknowledge trade-offs:** "This adds complexity in X but saves us in Y"
4. **Compare alternatives:** Show why other options don't address the real problem
5. **Show scalability thinking:** "At 10K rows this works; at 100M rows we'd consider [different approach]"
6. **Connect to production:** "In my role at [company], we handled this by..."

---

## Production-Grade Principles

These apply across all scenarios:

- **Set-based > Procedural:** SQL is optimized for bulk operations. Loops, cursors, row-by-row logic are performance killers. Default to set operations.
- **Constraints > Application Logic:** Database-level enforcement is safer than trusting app code. FKs, CHECK constraints, unique indexes should be your first instinct for integrity.
- **Audit Before Action:** Never revoke, delete, or restructure without understanding current state first. Measure before you optimize.
- **Non-Destructive First:** Prefer changes that are reversible. Partitioning, archiving, and role redesign are safer than deletion.
- **Audience Matters:** Your SQL, naming, and documentation should be written for the person *using* it, not just the person writing it.
- **Transparency Under Uncertainty:** When data is incomplete or assumptions are fuzzy (missing values, variable response counts), document explicitly. Don't silently "fix" data.

---

## Final Checklist: Before Submitting Your Answer

- [ ] Did I identify the *actual* bottleneck, not a symptom?
- [ ] Does my solution use set-based SQL, not procedural logic?
- [ ] If data integrity is involved, am I using constraints, not app validation?
- [ ] If performance is involved, did I consider query logic *before* indexes?
- [ ] If governance/maintenance is involved, did I prioritize clarity over cleverness?
- [ ] If missing data is involved, am I handling NULLs correctly and documenting counts?
- [ ] Did I explain *why* my answer is better, not just that it's better?
- [ ] Could I scale this approach to 10x, 100x the current data volume?

---

**Last Updated:** August 2026  
**Use Case:** Senior SQL & System Design Interview Prep  
**Level:** Senior Engineer / Principal (L5+)
