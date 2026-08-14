---
title: "SQL System Design & Problem-Solving Master Report (Part 1)"
date: 2026-08-14T00:00:00+05:30
draft: true
tags: ["sql", "system-design", "interview-prep"]
summary: "Senior-level SQL interview prep covering core gotchas and fundamentals, starting with NULL-handling traps."
---

# SQL System Design & Problem-Solving Master Report
## Senior-Level Interview Prep Guide

---

## 1. CORE SQL GOTCHAS & FUNDAMENTALS

### 1.1 NULL Handling — The Most Common Trap

**The Rule:** Comparisons with NULL always return UNKNOWN, never TRUE or FALSE. Rows filter out.

**Incorrect Approaches (These Fail):**
```sql
SELECT * FROM categories WHERE id <> NULL;  -- WRONG: never matches anything
SELECT * FROM categories WHERE id != NULL;  -- WRONG: same problem
SELECT * FROM categories WHERE id = NULL;   -- WRONG: never returns the NULL row
```

**Correct Approach:**
```sql
SELECT * FROM categories WHERE id IS NULL;       -- Correct: find NULLs
SELECT * FROM categories WHERE id IS NOT NULL;   -- Correct: exclude NULLs
```

**Interview Insight:** Any senior engineer should immediately catch NULL comparison errors. This filters when it shouldn't — data silently disappears from results.

---

### 1.2 DISTINCT Placement — Single Application Rule

**The Rule:** DISTINCT applies *once*, to the entire result row after SELECT. It cannot be repeated per column.

**Incorrect:**
```sql
SELECT DISTINCT a.id, DISTINCT b.id FROM customers a, customers b;
-- ERROR: DISTINCT can only appear once, right after SELECT
```

**Correct:**
```sql
SELECT DISTINCT a.id, b.id FROM customers a, customers b;
-- Applies to the row (a.id, b.id) as a unit
```

**Interview Insight:** Show understanding of WHERE the DISTINCT token sits in the execution model, not just that it removes duplicates.

---

### 1.3 Ambiguous Column References in Joins

**The Rule:** When a column name exists in both tables and isn't qualified, MySQL throws an ambiguous reference error.

**Errors Occur When:**
```sql
SELECT id, id FROM customers, customers;
-- ERROR: which 'id'? Both tables match.

SELECT id FROM customers, customers;
-- ERROR: unqualified 'id' matches both tables in FROM clause.

SELECT a.id, id FROM customers a, customers;
-- ERROR: the second 'id' is unqualified while 'id' exists in both tables.
```

**Solution: Always Qualify:**
```sql
SELECT a.id, b.id FROM customers a, customers b;
SELECT c.id FROM customers c;
```

**Interview Insight:** This catches junior developers. Senior engineers alias defensively from the start, even with single-table queries. Shows discipline and prevents silent bugs when schemas evolve.

---

## 2. AGGREGATION & GROUPING RULES

### 2.1 WHERE vs. HAVING Execution Order

**Critical Rule:** WHERE runs *before* grouping/aggregation. HAVING runs *after*.

**Consequence:** You cannot reference SELECT aliases or aggregates in WHERE.

**This Fails:**
```sql
SELECT customer_id, COUNT(*) AS transactions
FROM orders
WHERE transactions > 10  -- ERROR: 'transactions' doesn't exist yet (WHERE runs first)
GROUP BY customer_id;
```

**This Works:**
```sql
SELECT customer_id, COUNT(*) AS transactions
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 10;  -- Correct: HAVING runs after grouping
```

**Also Works:**
```sql
SELECT customer_id, COUNT(*) AS transactions
FROM orders
GROUP BY customer_id
HAVING transactions > 10;  -- MySQL allows alias reference in HAVING
```

**Interview Insight:** Explain the *execution order* to show deep understanding. Many seniors get this wrong under pressure.

---

### 2.2 GROUP BY with Non-Aggregated Columns

**MySQL Behavior (with ONLY_FULL_GROUP_BY OFF):**

```sql
SELECT customer_id, is_active, COUNT(*) AS count
FROM transactions
GROUP BY customer_id;
-- is_active is NOT in GROUP BY, but query runs (non-deterministic result per row)
```

This works syntactically but semantically is weak — unclear which `is_active` value you get per group.

**Best Practice:** Include all non-aggregated columns in GROUP BY or use aggregate functions:
```sql
SELECT customer_id, MAX(is_active) AS is_active, COUNT(*) AS count
FROM transactions
GROUP BY customer_id;
```

**Interview Insight:** Know the MySQL setting. Know when it's safe to break this rule (e.g., deterministic single-value columns you're confident about), but default to full correctness.

---

### 2.3 HAVING Can Reference Aliases (But Not All Databases Do)

**MySQL Allows:**
```sql
SELECT customer_id, MAX(is_active) AS is_active
FROM transactions
GROUP BY customer_id
HAVING is_active = 1;  -- MySQL permits this
```

**Safer (Portable):**
```sql
SELECT customer_id, MAX(is_active) AS is_active
FROM transactions
GROUP BY customer_id
HAVING MAX(is_active) = 1;  -- Works everywhere
```

**Interview Insight:** Mention MySQL's permissiveness, but show preference for standard SQL (re-compute the aggregate in HAVING). Demonstrates database portability awareness.

---

## 3. FORMATTING & DISPLAY ISSUES

### 3.1 ROUND() Returns Numeric Type — Trailing Zeros Lost

**The Problem:**
```sql
SELECT ROUND(amount, 2) FROM orders;
-- Result: 98.3  (not 98.30 even though ROUND to 2 decimals)
```

Numeric types don't store trailing zeros; they only matter for *display*.

**Solutions:**

**Option 1: FORMAT() — String with Formatting**
```sql
SELECT FORMAT(amount, 2) FROM orders;
-- Result: "98.30"  (adds thousands separators too: 1234.50 → "1,234.50")
```

**Option 2: CAST to DECIMAL — Keeps Type, Preserves Decimals**
```sql
SELECT CAST(amount AS DECIMAL(10,2)) FROM orders;
-- Result: 98.30  (numeric type, but many clients display trailing zeros)
```

**Interview Insight:** Show awareness of the difference between value and display. Know when to use a string (reports, client-facing) vs. keeping numeric type (further calculation).

---

### 3.2 ORDER BY After Formatting — Critical Bug

**The Bug:**
```sql
SELECT iban, FORMAT(amount, 2) AS amount
FROM balances
WHERE amount > 0 AND amount < 100
ORDER BY 2;  -- Sorting the formatted STRING, not the numeric column
```

**Result:** Alphabetic sort of strings like "99.58", "99.18", "98.92", "90.22", **"9.85"** ← Jumps here!

Because as strings: `"9.85"` < `"90.22"` (character-by-character: '9' vs '9', '.' vs '0', and '.' < '0' in ASCII).

**Fix: Sort Before Formatting**
```sql
SELECT iban, FORMAT(amount, 2) AS amount
FROM balances
WHERE amount > 0 AND amount < 100
ORDER BY amount DESC;  -- Sort the raw numeric column
```

**Interview Insight:** This is a real production bug. Sorting after casting to strings silently corrupts results. Always sort on the raw column; format in the SELECT list only for display.

---

## 4. QUERY OPTIMIZATION PATTERNS

### 4.1 CTEs for Readability & Maintainability

**Use Case:** Multi-level aggregation or deeply nested subqueries.

**Before (Hard to Maintain):**
```sql
SELECT outer_customer_id, AVG(monthly_spend)
FROM (
  SELECT customer_id AS outer_customer_id, SUM(spend) AS monthly_spend
  FROM (
    SELECT customer_id, MONTH(order_date) AS month, SUM(amount) AS spend
    FROM orders
    GROUP BY customer_id, MONTH(order_date)
  ) AS subq1
  GROUP BY customer_id
) AS subq2
GROUP BY outer_customer_id;
```

**After (Clear, Maintainable):**
```sql
WITH monthly_spend AS (
  SELECT customer_id, MONTH(order_date) AS month, SUM(amount) AS spend
  FROM orders
  GROUP BY customer_id, MONTH(order_date)
),
customer_avg AS (
  SELECT customer_id, AVG(spend) AS avg_monthly_spend
  FROM monthly_spend
  GROUP BY customer_id
)
SELECT * FROM customer_avg;
```

**Benefits:**
- Named steps make intent clear
- Easy to modify without parenthesis hell
- Often better query plan optimization
- Easier to test intermediate CTEs independently

**Interview Insight:** Refactoring ugly nested subqueries into CTEs shows seniority. Interviewers love this — it's professional code.

---

### 4.2 Joins Over Nested Subqueries (With Caveats)

**When It Works Well:**
Simple 1:N or N:1 joins often outperform nested subqueries at scale.

```sql
-- Subquery approach (can be slow)
SELECT * FROM customers c
WHERE c.id IN (
  SELECT customer_id FROM orders WHERE amount > 1000
);

-- Join approach (often faster)
SELECT DISTINCT c.* FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
WHERE o.amount > 1000;
```

**When Be Careful:**
Blindly converting aggregation subqueries to joins can cause **double-counting** if cardinality changes:

```sql
-- Correct (subquery guarantees 1:1)
SELECT c.id, 
  (SELECT COUNT(*) FROM orders WHERE customer_id = c.id) AS order_count
FROM customers c;

-- Risky (join without care — counts duplicate c.id rows)
SELECT c.id, COUNT(o.id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id;  -- Still works here with GROUP BY, but easy to get wrong
```

**Interview Insight:** Show nuance. Not all subqueries are bad; nested aggregations especially need care when converting. Suggest JOINs but justify *why* and acknowledge the cardinality risk.

---

### 4.3 When to Use ORDER BY Column Position

**Safe:**
```sql
SELECT id, name FROM categories ORDER BY 2;  -- Order by 'name'
```

**Risky (Maintenance):**
If someone later adds a new column in position 2, the sort changes silently:
```sql
-- Original
SELECT id, name FROM categories ORDER BY 2;

-- Refactored, unaware of the side effect
SELECT id, created_at, name FROM categories ORDER BY 2;  
-- Now sorts by created_at, not name!
```

**Best Practice:**
```sql
SELECT id, name FROM categories ORDER BY name DESC;  -- Explicit, safe
```

**Interview Insight:** Mention you know column-position sorting exists (older SQL style), but show preference for named columns in modern code. Demonstrate awareness of technical debt and maintenance burden.

---

## 5. SYSTEM DESIGN PATTERNS

### 5.1 Data Mart for Recurring Multi-Source Analytics

**Problem:**
- Multiple data sources (OLTP, OLAP, etc.)
- Complex joins taking 45+ minutes
- Performance degradation during peak hours
- Different teams concerned about cross-system query impact

**Solution: Purpose-Built Reconciliation Data Mart**

```sql
-- Scheduled load (e.g., nightly)
-- Extract and aggregate from both systems, pre-join
CREATE TABLE reconciliation_mart AS
SELECT 
  DATE_TRUNC(ot.transaction_date, MONTH) AS month,
  ot.product_id,
  SUM(ot.amount) AS oltp_revenue,
  SUM(dw.revenue) AS dw_revenue,
  SUM(ot.amount) - SUM(dw.revenue) AS discrepancy
FROM oltp_transactions ot
LEFT JOIN dw_aggregates dw 
  ON ot.product_id = dw.product_id 
  AND DATE_TRUNC(ot.transaction_date, MONTH) = dw.month
GROUP BY DATE_TRUNC(ot.transaction_date, MONTH), ot.product_id;

-- Then report queries run lightning-fast against the mart
SELECT * FROM reconciliation_mart WHERE discrepancy != 0;
```

**Benefits:**
- Report queries complete in seconds, not 45 minutes
- OLTP/OLAP systems only touched during controlled scheduled extracts
- Single, optimized structure for the specific use case
- Data quality layer can validate/reconcile before mart population

**Interview Insight:** Shows understanding of data warehouse patterns and decoupling. Demonstrates thinking beyond "just write a good query" to architectural solutions.

---

### 5.2 Table Partitioning for Time-Series Operational Data

**Problem:**
- Monitoring table grows 50GB/month (18 months = 900GB)
- Real-time queries on recent data slow during peak hours
- Policy: keep 3 months readily accessible, 12 months for trend analysis, older archived

**Solution: Monthly Partition + Tablespaces**

```sql
CREATE TABLE application_logs (
  log_id BIGINT PRIMARY KEY,
  timestamp DATETIME,
  error_code INT,
  message TEXT,
  created_at DATETIME
)
PARTITION BY RANGE (YEAR(created_at) * 100 + MONTH(created_at)) (
  PARTITION p202401 VALUES LESS THAN (202402),
  PARTITION p202402 VALUES LESS THAN (202403),
  PARTITION p202403 VALUES LESS THAN (202404),
  -- ... continues
  PARTITION p202406 VALUES LESS THAN (202407),  -- Recent (fast storage)
  PARTITION p_archive VALUES LESS THAN MAXVALUE  -- Older (slow storage)
);

-- Route partitions to tablespaces
ALTER TABLE application_logs MODIFY PARTITION p202401 
  DATA DIRECTORY = '/archive_tablespace/';  -- Cheaper storage
```

**Query Impact:**
```sql
-- Recent data query (scans only June partition, ~50GB, fast)
SELECT * FROM application_logs 
WHERE created_at BETWEEN '2024-06-01' AND '2024-06-30' 
  AND error_code = 500;

-- Trend query (scans 3 partitions, ~150GB, acceptable)
SELECT MONTH(created_at), COUNT(*) FROM application_logs
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
GROUP BY MONTH(created_at);
```

**Benefits:**
- Peak-hour queries on recent data stay fast (partition pruning)
- Aged data moves to cheaper storage, not deleted (compliance)
- New partitions auto-added; old ones aged out systematically
- Scales indefinitely with predictable performance

**Interview Insight:** Show awareness of operational vs. analytical workloads. Demonstrate thinking about retention policies, compliance, and cost. Partitioning is a senior-level tool.

---

### 5.3 Staging Area + Controlled Extraction

**Pattern:** Intermediate buffer for cross-system data loads, decouples source systems.

```sql
-- Step 1: Controlled extract from OLTP (off-peak, scheduled)
-- Run once/day, 2 AM
INSERT INTO staging.oltp_snapshot (extracted_at, data)
SELECT NOW(), json_object(
  'transaction_id', id,
  'customer_id', customer_id,
  'amount', amount,
  'date', transaction_date
)
FROM oltp_transactions
WHERE transaction_date >= DATE_SUB(NOW(), INTERVAL 1 DAY);

-- Step 2: Validation layer (check for nulls, duplicates, date ranges)
-- Run immediately after extraction
SELECT COUNT(DISTINCT transaction_id), COUNT(*) FROM staging.oltp_snapshot
WHERE extracted_at = CURDATE();
-- Alert if cardinality looks wrong

-- Step 3: Load into data mart (only after validation passes)
INSERT INTO reconciliation_mart (...)
SELECT ... FROM staging.oltp_snapshot WHERE extracted_at = CURDATE();
```

**Interview Insight:** Staging area is a common enterprise pattern. Shows awareness of data quality, lineage, and decoupling live systems from analytical workloads.

---

## 6. DECISION FRAMEWORK FOR SENIOR INTERVIEWS

### 6.1 When You See a Problem, Ask These Questions

| Problem Type | First Question | Likely Approach |
|---|---|---|
| "Make query faster" | Real-time or batch? | Index, partition, or data mart |
| "Reduce storage" | Is data still needed? | Compress, archive, or partition to cheaper storage |
| "Handle multiple sources" | How often does reporting run? | Staging area + mart, or federation |
| "Ugly nested query" | Is this for production or report? | Refactor to CTE or replace with JOIN (with cardinality check) |
| "Report takes 45 mins" | Who queries it? When? | Dedicated data mart, scheduled load |
| "Schema change breaks queries" | How often does this happen? | Better documentation, CI/CD validation, or alias defensively |

---

### 6.2 Red Flags & Gotchas to Catch in Code Reviews

1. **Comparing to NULL** → Use IS NULL / IS NOT NULL
2. **ORDER BY after FORMAT()** → Sort the raw column; format in SELECT
3. **Blindly converting subqueries to JOINs** → Check cardinality, especially with aggregates
4. **GROUP BY with non-aggregated, non-functional columns** → Add to GROUP BY or aggregate
5. **WHERE clause filtering on aliases** → Move logic to HAVING or pre-filter in FROM
6. **Ambiguous column names (unqualified)** → Qualify all columns in multi-table queries
7. **Single table query running 45+ minutes** → Partition, index, or redesign with staging
8. **Hardcoded DISTINCT without understanding scope** → Justify; it applies to the entire row

---

## 7. INTERVIEW TALKING POINTS

### Opening Statement (When Asked to Optimize/Design)
> "Before I propose a solution, I'd ask: What's the current performance bottleneck? Is this a real-time query or a batch report? Who runs it, when, and how often? Are there retention/compliance requirements? That context drives whether I'm optimizing the query itself, refactoring to CTEs, introducing indices, building a data mart, or partitioning the table. There's rarely one answer."

### On Refactoring Nested Subqueries
> "I'd first convert deeply nested subqueries to CTEs for readability — that's often a productivity win before touching performance. Then, if it's still slow, I'd check if we can use JOINs instead. But I'd be careful: if there are aggregates, converting to a JOIN can cause double-counting if I'm not accounting for cardinality correctly. A DISTINCT or GROUP BY can fix that, but I'd validate the results match the original query."

### On Partitioning & Archival
> "Partitioning solves multiple problems at once: it keeps recent data queries fast (partition pruning), allows older data to move to cheaper storage without deletion, and lets you implement retention policies systematically. The trade-off is administrative overhead — you need a job to create new partitions and age out old ones. For a 50GB/month growth rate, this pays for itself quickly."

### On Data Marts
> "If the same complex query runs repeatedly against live systems, a data mart is usually the answer. You extract and pre-aggregate data on a schedule, then users query the mart. It decouples your analytics workload from production, gives you a validation layer, and makes every subsequent report query instant. The cost is latency — data is as fresh as your last load — but for most reporting, daily or hourly loads are fine."

---

## 8. QUICK REFERENCE CHECKLIST

**Before writing any query:**
- [ ] All columns in GROUP BY or aggregated?
- [ ] WHERE vs. HAVING logic separated correctly?
- [ ] NULL handling explicit (IS NULL / IS NOT NULL)?
- [ ] Joins qualified (table.column)?
- [ ] ORDER BY on raw columns, not formatted/casted?
- [ ] Result cardinality as expected?

**Before proposing optimization:**
- [ ] Real-time or batch?
- [ ] Current bottleneck: query plan, storage, or architecture?
- [ ] Retention/compliance requirements?
- [ ] Cross-system or single-source?
- [ ] Does a data mart make sense?
- [ ] Should this be partitioned?

**Before refactoring:**
- [ ] Does this improve readability?
- [ ] Does this hurt performance (test first)?
- [ ] Does this maintain correctness (especially cardinality)?
- [ ] Will future maintainers understand this?

---

## 9. PRACTICE SCENARIOS

### Scenario A: Slow Daily Report
**Given:** A daily financial reconciliation report joining `sales` (OLTP), `costs` (OLAP), and `inventory` tables. Report takes 45 minutes; users complain during 9 AM daily standup.

**Senior Answer:**
1. First, I'd check: Is the report truly needed to be real-time, or can it run at 6 AM?
2. If it can run off-hours, build a data mart: extract, validate, and pre-join the three tables nightly. Report queries then run in seconds.
3. If real-time is non-negotiable, add indices on the join keys (sales.product_id, costs.product_id, inventory.product_id) and ensure statistics are fresh.
4. Avoid the three-table join in the mart if possible; instead, pre-aggregate from sales and costs separately, then LEFT JOIN to inventory only if inventory levels are truly needed (they often aren't for profit reporting).

---

### Scenario B: NULL Handling Bug
**Given:** A query filtering active customers returns fewer rows than expected after a code change.

**Buggy Code:**
```sql
SELECT * FROM customers WHERE status != 'inactive';
```

**Senior Answer:**
If any `status` values are NULL, they're silently filtered out (NULL != 'inactive' returns UNKNOWN, not TRUE). Fix with:
```sql
SELECT * FROM customers WHERE status != 'inactive' OR status IS NULL;
-- Or
SELECT * FROM customers WHERE COALESCE(status, '') != 'inactive';
```

Then ask: What should NULL status mean? Incomplete signup? And should the upstream schema default status to something explicit instead?

---

### Scenario C: Partitioning Strategy
**Given:** A logs table with 100GB/day growth; compliance requires 90 days retention; queries are usually on the last 7 days.

**Senior Answer:**
Daily partitions (not monthly, too granular for this scale). Set an automated job to create tomorrow's partition and drop partitions older than 90 days. Route recent partitions to fast SSD storage, older partitions to cheaper HDD/cloud storage. Query performance stays flat as data grows; no one notices the 90-day retention boundary operationally.

---

## 10. FINAL REMINDERS

- **Correctness first, optimization second.** A slow query is better than a fast wrong one.
- **Ask clarifying questions.** The best senior engineers ask before proposing.
- **Explain trade-offs.** Partitioning, data marts, and CTEs all have costs. Own them.
- **Test your assumptions.** Don't guess at cardinality or performance; measure.
- **Think operationally.** How will the DBA monitor this? Who on-calls? Will it scale?
- **Show your work.** Interviewers care as much about your reasoning as your answer.

---

**Last Updated:** August 2026  
**Prepared for:** Senior ML Engineer / Senior Data Engineer Interviews  
**Focus:** AdTech, Production Systems, and Scalability
