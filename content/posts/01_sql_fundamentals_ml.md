---
title: "SQL Fundamentals for Senior ML Scientists (v1)"
date: 2026-07-19T00:00:00+05:30
draft: false
tags: ["sql", "interview-prep"]
summary: "A reasoning-first SQL refresher for ML interviews, prioritizing why a query works over syntax memorization."
---

# SQL Fundamentals for Senior ML Scientists

> Prioritises reasoning over syntax. Know the *why*, defend the *how*.

---

## 1. WHAT IS A DATABASE?

A database is a file system that understands **relationships** and enforces **consistency** — not a folder of CSVs.

**Organisation hierarchy**:
```
Database → Schema → Table / View / Index / UDF
```

**Why databases exist (ACID)**:

| Property | Intuition | Rule | Example |
|----------|-----------|------|---------|
| **Atomicity** | All or nothing | Every step must succeed, or the whole transaction rolls back | Moving money: debit A + credit B. If credit fails, debit is cancelled — no money vanishes |
| **Consistency** | Valid states only | A transaction moves the DB from one valid state to another — no constraint is broken mid-way | Total money in the bank is identical before and after a transfer |
| **Isolation** | No interference | Concurrent transactions run as if sequential — they can't see each other's in-progress changes | Two users buying the last seat simultaneously: only one succeeds, the other sees "sold out" |
| **Durability** | Permanent save | Once committed, data survives crashes, power cuts, restarts | Your completed transfer still exists when the server comes back after a blackout |

**Structured vs Unstructured**: SQL = rows, columns, schema, fast JOINs. NoSQL = flexible, no native JOINs, eventual consistency. You use SQL because features are relational, ACID matters, and JOINs beat Python merges.

---

## 2. TYPES OF SQL

| Type | Purpose | Commands | Your Context |
|------|---------|----------|-------------|
| **DDL** (Definition) | Define schema structure | `CREATE`, `DROP`, `ALTER TABLE` | Written by DevOps via Flyway. Auto-commits — no rollback. |
| **DML** (Manipulation) | Read/write data | `SELECT`, `INSERT`, `UPDATE`, `DELETE` | 99% of your work. Can rollback. |
| **DCL** (Control) | Manage permissions | `GRANT`, `REVOKE` | Why "permission denied" errors happen |
| **TCL** (Transaction) | Transaction lifecycle | `BEGIN`, `COMMIT`, `ROLLBACK` | Wraps ETL pipelines for atomicity |

---

## 3. OLTP vs OLAP: Why Vertica Exists

**Row-store (OLTP — PostgreSQL)**: Stores data row-by-row. Fast for single-row lookups, slow for analytics — reads all 50 columns even when you need 3.

**Column-store (OLAP — Vertica)**: Stores data column-by-column. Reads only the columns your query touches.

```
Query: SELECT user_id, clicks FROM fact_impressions WHERE log_time > '2024-01-01'

PostgreSQL: reads full row (all 50 columns) × 1B rows = 4TB I/O
Vertica:    reads 3 columns × 1B rows, compressed = ~500MB I/O
```

**Why Vertica is 60× faster at 1B rows**:

| Mechanism | What It Does | Speedup |
|-----------|-------------|---------|
| Column storage | Read only needed columns | 8000× less I/O |
| Compression (RLE, dict, bit-pack) | Same-type columns compress 8× | Fits in cache |
| CPU cache-friendly | Sequential column reads = 95% cache hits vs 70% | 10× |
| Vectorisation (SIMD) | Process 10K rows/batch, not 1 row/cycle | 4–8× |
| Projections | Pre-materialised column subsets | 15–50× on hot queries |
| Parallelisation | Auto-distributed across CPUs | 4× on quad-core |

**When to use which**:

| Workload | Tool | Why |
|----------|------|-----|
| Transactions, real-time updates | OLTP (PostgreSQL) | Single-row access |
| Analytics, feature engineering, 1B rows | OLAP (Vertica, Snowflake) | Column-scan |

**Vertica vs Snowflake**: Same column-store model; Vertica is on-premise (fixed cost), Snowflake is cloud-native (pay per compute hour). SQL is 95% identical between them.

---

## 4. SCHEMA DESIGN: Why fact\_, dim\_, agg\_

**3NF (Normalisation)**: Eliminate redundancy by splitting tables. One source of truth.

```
BAD:  campaign_id=101, budget=10000 repeated on 1M impression rows  → update nightmare
GOOD: dim_campaigns(campaign_id, budget) stored once; fact_impressions references campaign_id
```

**Star Schema**: One fact table (events/metrics) + dimension tables (attributes). One JOIN per dimension — fast, predictable.

```
fact_impressions(user_id, campaign_id, log_time, clicks)
     ↓                     ↓
dim_users(user_id, country)  dim_campaigns(campaign_id, name, budget)
```

**Snowflake Schema**: Like star, but dimensions are further normalised (dimensions join to sub-dimensions). More JOINs, less storage, slower reads.

**Your Adform design** (`dsp.*`, `tpas.*`, `train.*`):

| Table Type | Purpose | Example |
|-----------|---------|---------|
| `fact_*` | Raw events (granular, append-only) | `dsp.fact_impressions_full` |
| `dim_*` | Attributes (slowly changing) | `dsp.dim_placements` |
| `agg_*` | Pre-computed aggregations (nightly) | `agg_publisher_ctr` |

**Why separate schemas per domain**: DSP team owns `dsp.*`, fraud team owns `tpas.*` — no coordination. Different retention policies (RTB = 90 days, fraud = 7 years). Different backup strategies per schema.

**Why `agg_*` tables**: Pre-compute once nightly, query 1000× fast. Without them, every training run re-aggregates 1B rows.

---

## 5. VIEWS VS TABLES

**Core mental model**:
- **Table**: Data physically stored on disk. Pay storage, get fast reads.
- **View**: Saved query definition. Computed on-the-fly — always fresh, no storage cost.
- **Materialized view**: Computed once and stored. Refreshed on schedule. Fast reads + acceptable staleness.

**Decision**:

| Scenario | Choice | Why |
|----------|--------|-----|
| Fact/dimension data you own | Table | Store once, read 1000× |
| Simple filter, queried rarely | View | No storage, always accurate |
| Expensive aggregation, queried frequently | Materialized view | Pre-compute off-peak, read fast |
| Real-time bid data (updates every ms) | Table | Materialized views can't keep pace |

**Your `agg_*` tables** = materialized views refreshed nightly by Flyway. Staleness acceptable (yesterday's features fine for morning training).

```sql
CREATE MATERIALIZED VIEW agg_user_fraud_score AS
SELECT user_id,
       AVG(amount)    AS avg_txn,
       STDDEV(amount) AS stddev_txn,
       COUNT(*)       AS txn_count
FROM transactions
WHERE timestamp > CURRENT_DATE - INTERVAL '90 days'
GROUP BY user_id;

REFRESH MATERIALIZED VIEW agg_user_fraud_score;  -- runs nightly
```

---

## 6. EXECUTION SEQUENCE

You write SQL in one order; it executes in a completely different order.

```
Write order:   SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY
Execute order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

**The key trap — WHERE vs HAVING**:

```sql
-- WRONG: SUM doesn't exist yet when WHERE runs
SELECT user_id, SUM(clicks)
FROM impressions
WHERE SUM(clicks) > 100   -- ❌ Error: aggregate in WHERE
GROUP BY user_id;

-- RIGHT: HAVING runs after GROUP BY
SELECT user_id, SUM(clicks)
FROM impressions
GROUP BY user_id
HAVING SUM(clicks) > 100;  -- ✅
```

**Rule**: `WHERE` filters **rows** (before grouping). `HAVING` filters **groups** (after aggregating).

**Your real pattern** (from Adform queries):
```sql
SELECT DATE_TRUNC('HOUR', log_time) AS log_hour,
       SUM(CASE WHEN is_dsp  THEN 1 ELSE 0 END) AS rtb_impressions,
       SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count
FROM tpas.fact_all_impressions_full
WHERE log_time >= '2024-04-23 00:00:00'   -- WHERE runs first, filters rows
  AND NOT is_ppas
GROUP BY 1;                                -- then GROUP BY aggregates
```

---

## 7. INDEXING

**Intuition**: A book index — jump directly to the relevant page rather than reading every page.

**Three types**:

| Type | Best For | Used In |
|------|---------|---------|
| **B-Tree** | Equality + range queries (`user_id = 5`, `ts > date`) | OLTP (PostgreSQL) |
| **Hash** | Exact match only — no ranges | Rare |
| **Columnar** | Column-scan analytics — the column *is* the index | Vertica (RLE, bit-vectors) |

**When to index**:

| Condition | Index? | Reason |
|-----------|--------|--------|
| Frequently filtered, high cardinality (`user_id`, `timestamp`) | ✅ Yes | Selectivity benefit |
| JOIN column | ✅ Yes | Join condition scanned repeatedly |
| Low cardinality (gender: M/F) | ❌ No | Most queries return most rows anyway |
| Table < 100M rows | ❌ Often no | Full scan is fast enough |

**Vertica note**: Column encoding replaces traditional indexes. Proper column selection and projections matter more than explicit indexing.

---

## 8. ETL & HOW YOUR TABLES GET POPULATED

**ETL flow**: Extract (raw logs) → Transform (aggregate, clean) → Load (write to Vertica)

**Your setup**:

```
Real-time events → fact tables (dsp.fact_impressions_full, tpas.fact_all_impressions_full)
Nightly Flyway job → aggregations written to agg_* tables
ML pipeline → reads agg_* (fast, pre-computed)
```

**Flyway**: Versioned SQL migration tool. Tracks which scripts ran, prevents double-execution, enables rollback. Your schema changes (`CREATE TABLE`, `ALTER TABLE`) live in versioned Flyway scripts — not ad-hoc DDL.

**Nightly aggregation pattern** (from your queries):
```sql
DROP TABLE IF EXISTS temp.agg_hourly_impressions;
CREATE TABLE temp.agg_hourly_impressions AS
SELECT DATE_TRUNC('HOUR', log_time)                            AS log_hour,
       SUM(CASE WHEN is_dsp   THEN 1 ELSE 0 END)              AS dsp_impressions,
       SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)              AS fraud_count,
       COUNT(DISTINCT cookie_id)                               AS unique_users
FROM tpas.fact_all_impressions_full
WHERE log_time >= CURRENT_DATE - INTERVAL '1 day'
GROUP BY 1;
```

**Why fixed intervals**: Predictable availability (features ready by 3 AM), compute-once efficiency, atomic batch load (no partial data).

---

## 9. USER-DEFINED FUNCTIONS (UDFs)

Custom logic you define in SQL. Use sparingly.

| Type | Returns | When to Use |
|------|---------|-------------|
| **Scalar** | Single value per row | Reusable transform (format phone, decode flag) |
| **Table-valued** | Set of rows | Encapsulate complex multi-row logic |

**Use UDFs when**: Logic is complex AND reused across 3+ queries AND performance is not critical.

**Avoid UDFs when**: Called row-by-row on large tables (1M calls = no vectorisation = slow). Inline the logic instead:

```sql
-- SLOW: UDF called 1M times
SELECT user_id, fraud_score(user_id, amount) FROM transactions;

-- FAST: Inline logic, optimizer can vectorise
SELECT user_id,
       (CASE WHEN amount > 10000 THEN 50 ELSE 0 END +
        CASE WHEN is_new_location  THEN 30 ELSE 0 END) AS fraud_score
FROM transactions;
```

---

## 10. CONNECTING FROM PYTHON

**Connection pooling** (always use — opens connections once, reuses them):
```python
from sqlalchemy import create_engine
engine = create_engine("vertica+pyodbc://user:pass@host/db", pool_size=5)
```

**Three fetch patterns**:

```python
# Small result (<1M rows) — load all at once
df = pd.read_sql("SELECT ...", engine)

# Large result — stream in chunks
for chunk in pd.read_sql("SELECT ...", engine, chunksize=50_000):
    process(chunk)

# Raw cursor — maximum control
with engine.connect() as conn:
    result = conn.execute("SELECT ...")
```

**Rule**: Always filter in SQL before loading to Python. Never `SELECT *` then filter in Pandas on 1B rows.

---

## 11. SQL vs PANDAS: WHERE TO COMPUTE

| Computation Type | Tool | Why |
|-----------------|------|-----|
| Aggregation (1M → 1K rows) | **SQL** | DB parallelises, compressed, no RAM cost |
| Row-level transforms (1M → 1M rows) | **Pandas** | Custom logic, ML libraries available |
| Aggregate then enrich | **SQL → Pandas** | DB does heavy lifting, Pandas does finesse |

**Examples from your work**:

```sql
-- SQL: aggregate 1M → 24 rows
SELECT DATE_TRUNC('HOUR', log_time)                             AS log_hour,
       SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)               AS fraud_count,
       COUNT(DISTINCT cookie_id)                                AS unique_users
FROM tpas.fact_all_impressions_full
WHERE log_time >= '2024-04-23' AND log_time < '2024-04-24'
GROUP BY 1;
```

```python
# Pandas: enrich 24 rows with ML features
df['fraud_rate']       = df['fraud_count'] / df['unique_users']
df['fraud_rate_pct']   = df['fraud_rate'].rank(pct=True)
df['fraud_7d_avg']     = df['fraud_rate'].rolling(7, min_periods=1).mean()
```

**Default**: SQL aggregates. Pandas enriches. Never pull 1M raw rows to Pandas when a `GROUP BY` gives you 1K.

---

## 12. OUTPUT FORMATS

| Format | Use When | Key Trait |
|--------|---------|-----------|
| **CSV** | Sharing with non-technical users, < 100MB | Universal, human-readable; slow I/O, no types |
| **Feather** | Feature store, repeated Python reads | Sub-second read, native types; Python-only, no compression |
| **Parquet** | Archive, data lake, 1B+ rows | 4–8× compressed, Spark-native, standardised |
| **XLSX** | Business reports, stakeholders | Multi-sheet, formatted; 1M row limit |

```python
df.to_csv('out.csv')                          # CSV
df.to_feather('out.feather')                  # Feather
df.to_parquet('out.parquet', compression='snappy')  # Parquet
df.to_excel('out.xlsx')                       # XLSX
```

**Default for 1M+ rows**: Feather (Python pipeline) or Parquet (archive/distributed).

---

## 13. VERTICA & SNOWFLAKE: QUICK REFERENCE

| Property | Vertica | Snowflake |
|----------|---------|-----------|
| **Deployment** | On-premise | Cloud (AWS/Azure/GCP) |
| **Cost model** | Fixed infrastructure | Pay-per-compute-hour |
| **Storage** | Column-oriented | Column-oriented |
| **Best for** | On-prem analytics at scale | Cloud-native, elastic workloads |
| **SQL compatibility** | Standard + Vertica extensions | Standard + Snowflake extensions |

**SQL is 95% identical**. Differences surface only in UDF syntax, materialized view refresh commands, and date function variants.

**Your stack**: Vertica (on-prem, `dsp.*` / `tpas.*` / `train.*` schemas) + Snowflake (cloud reporting). Same query logic, different connection strings.

---

## DECISION CHEAT SHEET

| Question | Answer |
|---------|--------|
| WHERE vs HAVING? | WHERE = filter rows (before GROUP). HAVING = filter groups (after GROUP). |
| Table vs View vs Mat.View? | Data you own = table. Cheap filter = view. Expensive agg queried often = mat.view. |
| SQL vs Pandas? | Aggregate in SQL. Transform row-by-row in Pandas. |
| Index or not? | High-cardinality filter/JOIN column on >100M rows = yes. Otherwise = no. |
| Vertica vs PostgreSQL? | Analytics at 1B rows = Vertica. Transactions/single-row = PostgreSQL. |
| Which output format? | CSV (share), Feather (fast Python), Parquet (archive), XLSX (business). |

