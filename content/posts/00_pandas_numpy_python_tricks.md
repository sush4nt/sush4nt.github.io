---
title: "Pandas, NumPy, Python Tricks: Interview-Ready Reference"
date: 2026-07-19T00:00:00+05:30
draft: true
tags: ["python", "pandas", "numpy", "interview-prep"]
summary: "A running set of toy datasets and idiomatic pandas/NumPy/Python snippets for common data manipulation interview tasks."
---

# Pandas, NumPy, Python Tricks: Interview-Ready Reference

## Toy Datasets (Running Examples)

### Loan Default Classification (10 samples)
```python
import pandas as pd
import numpy as np

# Loan dataset: predict default (1) vs no default (0)
loans = pd.DataFrame({
    'customer_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'age': [25, 35, 45, 28, 52, 31, 48, 26, 38, 42],
    'income': [30000, 45000, 65000, 35000, 75000, 40000, 60000, 32000, 50000, 70000],
    'credit_score': [650, 720, 780, 680, 800, 700, 750, 640, 730, 790],
    'loan_amount': [5000, 10000, 20000, 8000, 30000, 12000, 25000, 6000, 15000, 28000],
    'default': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
})

# House Price Regression (10 samples)
houses = pd.DataFrame({
    'house_id': range(1001, 1011),
    'sqft': [1200, 1500, 2000, 1800, 2500, 1300, 1900, 1100, 2200, 1600],
    'bedrooms': [2, 3, 4, 3, 5, 2, 4, 2, 4, 3],
    'age_years': [10, 5, 15, 8, 3, 20, 7, 12, 6, 9],
    'price': [200000, 280000, 350000, 320000, 450000, 220000, 330000, 180000, 380000, 280000]
})
```

---

## PART 1: Python Fundamentals (Comprehensions & Tricks)

### List Comprehensions

**Core Pattern:**
```python
[expression for item in iterable if condition]
```

**Example 1: Filter and transform**
```python
# Get customers over 30 with income > 40000
high_earners = [
    f"Customer {cid}: ${inc}" 
    for cid, inc in zip(loans['customer_id'], loans['income']) 
    if inc > 40000
]
# → ['Customer 102: $45000', 'Customer 103: $65000', ...]
```

**Example 2: Nested structure**
```python
# Create list of (customer_id, income, age) for high earners
high_earner_tuples = [
    (row['customer_id'], row['income'], row['age']) 
    for _, row in loans[loans['income'] > 40000].iterrows()
]
# Watch out: iterrows() is slow on large DataFrames; use later for scaling.
```

**Tricky: Nested comprehension**
```python
# Flatten nested lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [item for row in matrix for item in row]
# → [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Think: outer loop (row), inner loop (item)
```

### Dictionary Comprehensions

**Core Pattern:**
```python
{key_expr: value_expr for item in iterable if condition}
```

**Example 1: Create lookup dict**
```python
# Map customer_id → credit_score
credit_by_id = {
    row['customer_id']: row['credit_score'] 
    for _, row in loans.iterrows()
}
# → {101: 650, 102: 720, ...}
```

**Example 2: Group aggregation as dict**
```python
# Count defaults by income bucket
income_buckets = {
    'low': (0, 40000),
    'mid': (40000, 60000),
    'high': (60000, float('inf'))
}
defaults_by_bucket = {
    bucket: loans[
        (loans['income'] >= income_buckets[bucket][0]) & 
        (loans['income'] < income_buckets[bucket][1])
    ]['default'].sum()
    for bucket in income_buckets
}
# → {'low': 2, 'mid': 1, 'high': 0}
# Better approach: use pandas cut() [covered later]
```

**Tricky: Dict from two lists**
```python
# Zip two lists into a dict
ids = [101, 102, 103]
scores = [650, 720, 780]
id_to_score = {k: v for k, v in zip(ids, scores)}
# → {101: 650, 102: 720, 103: 780}
```

### Dictionary Sorting

**Sort by keys:**
```python
# Unsorted
credit_by_id = {103: 780, 101: 650, 102: 720}

# Sort by keys (ascending)
sorted_by_key = dict(sorted(credit_by_id.items()))
# → {101: 650, 102: 720, 103: 780}

# Sort by keys (descending)
sorted_by_key_desc = dict(sorted(credit_by_id.items(), reverse=True))
# → {103: 780, 102: 720, 101: 650}
```

**Sort by values:**
```python
# Sort by values (ascending)
sorted_by_value = dict(sorted(credit_by_id.items(), key=lambda x: x[1]))
# → {101: 650, 102: 720, 103: 780}

# Sort by values (descending)
sorted_by_value_desc = dict(sorted(credit_by_id.items(), key=lambda x: x[1], reverse=True))
# → {103: 780, 102: 720, 101: 650}
```

**Real example: Top customers by loan amount**
```python
customer_loans = {
    'alice': 5000,
    'bob': 20000,
    'charlie': 15000,
    'diana': 30000
}

# Get top 3 by loan amount
top_3 = dict(
    sorted(customer_loans.items(), key=lambda x: x[1], reverse=True)[:3]
)
# → {'diana': 30000, 'bob': 20000, 'charlie': 15000}
```

**Tricky: Sort nested dict**
```python
# Dict of customer → {age, income}
customers = {
    101: {'age': 25, 'income': 30000},
    102: {'age': 35, 'income': 45000},
    103: {'age': 28, 'income': 35000}
}

# Sort by income (value of nested dict)
sorted_by_income = dict(
    sorted(customers.items(), key=lambda x: x[1]['income'], reverse=True)
)
# → {102: {...}, 103: {...}, 101: {...}}
```

**Interview tip:**
> "When sorting a dict by values, I use `sorted(dict.items(), key=...)`. Python 3.7+ preserves insertion order in dicts, so I can convert back with `dict()`. For large datasets, I'd use Pandas instead—it's faster and more readable."

### Generator Expressions

**When to use:** Large datasets, one-pass consumption, memory efficiency.

**Core Pattern:**
```python
(expression for item in iterable if condition)  # Note: parens, not brackets
```

**Example 1: Lazy evaluation**
```python
# Don't create full list; iterate as needed
high_income_gen = (
    inc for inc in loans['income'] if inc > 40000
)
for income in high_income_gen:
    print(income)  # Processes one at a time
```

**Example 2: Sum/average without materializing**
```python
avg_income = sum(
    inc for inc in loans['income'] if inc > 40000
) / len([inc for inc in loans['income'] if inc > 40000])
# Lazy version (avoid double-counting):
filtered = (inc for inc in loans['income'] if inc > 40000)
incomes = list(filtered)
avg = sum(incomes) / len(incomes)
```

**Interview takeaway:** 
> "Generators are memory-efficient for streaming data. If I'm processing a large file line-by-line, I'd use a generator expression rather than loading everything into memory."

---

## PART 2: NumPy Essentials

### Arrays & Basic Operations

**Array creation:**
```python
import numpy as np

# From Python list
arr = np.array([1, 2, 3, 4, 5])

# Ranges
arr = np.arange(0, 10, 2)  # → [0, 2, 4, 6, 8]

# Linspace (useful for interpolation)
arr = np.linspace(0, 10, 5)  # → [0., 2.5, 5., 7.5, 10.]

# Zeros, ones
zeros = np.zeros(5)  # → [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))  # → [[1. 1. 1.] [1. 1. 1.]]

# Random
rand = np.random.rand(5)  # → [0.123, 0.456, ...] (uniform [0,1))
normal = np.random.normal(0, 1, 5)  # → normal distribution
```

**Vectorized operations (NO loops):**
```python
# Element-wise arithmetic
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = arr1 + arr2  # → [5, 7, 9]  (NOT a loop)
result = arr1 * arr2  # → [4, 10, 18]

# Comparison (returns boolean array)
mask = arr1 > 2  # → [False, False, True]
filtered = arr1[mask]  # → [3]

# Aggregate functions
np.sum(arr1)  # → 6
np.mean(arr1)  # → 2.0
np.std(arr1)  # → 0.816...
np.min(arr1), np.max(arr1)  # → (1, 3)
```

**Why NumPy is fast:**
- Operations are **vectorized** (compiled C loops, not Python)
- Example: 1M element addition: NumPy ~0.001s vs Python loop ~1s

### Broadcasting (The Mind-Bender)

**What is it:** Automatically aligning arrays of different shapes for operations.

**Example 1: Add scalar to array**
```python
arr = np.array([1, 2, 3])
result = arr + 10  # Broadcasting adds 10 to each element
# → [11, 12, 13]
```

**Example 2: 2D + 1D**
```python
# Feature matrix: 10 samples, 3 features
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
# Shape: (3, 3)

# Subtract mean of each column
col_means = np.array([4, 5, 6])  # Shape: (3,)
X_centered = X - col_means  # Broadcasting aligns shapes
# → [[−3, −3, −3], [0, 0, 0], [3, 3, 3]]
```

**Example 3: When broadcasting fails**
```python
a = np.array([[1, 2], [3, 4]])  # Shape: (2, 2)
b = np.array([1, 2, 3])  # Shape: (3,)
result = a + b  # ❌ ValueError: operands could not be broadcast
```

**Broadcasting rules:**
- Trailing dimensions must match OR one must be 1
- Shape (2, 3) + (3,) → OK (align on right)
- Shape (2, 3) + (2, 1) → OK (1 broadcasts)
- Shape (2, 3) + (1, 3) → OK (2 broadcasts)

### Reshaping & Indexing

**Reshape:**
```python
arr = np.arange(12)  # [0, 1, 2, ..., 11], shape (12,)
reshaped = arr.reshape(3, 4)
# → [[0, 1, 2, 3],
#    [4, 5, 6, 7],
#    [8, 9, 10, 11]]

# Flatten back
flat = reshaped.flatten()  # → [0, 1, 2, ..., 11]
```

**Advanced indexing:**
```python
arr = np.arange(20).reshape(4, 5)
arr[0, :]  # First row: [0, 1, 2, 3, 4]
arr[:, 2]  # Third column: [2, 7, 12, 17]
arr[1:3, 1:4]  # 2×3 subarray

# Boolean indexing
mask = arr > 10
arr[mask]  # → [11, 12, 13, 14, 15, 16, 17, 18, 19]
```

### Common Pitfalls

**Pitfall 1: In-place operations**
```python
arr = np.array([1, 2, 3])
arr += 10  # Modifies original
arr = arr + 10  # Creates new array (safer)
```

**Pitfall 2: Copying vs. views**
```python
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]  # This is a VIEW, not a copy
view[0] = 999
print(original)  # → [1, 999, 3, 4, 5] (modified!)

# To avoid:
copy = original[1:4].copy()
copy[0] = 999
print(original)  # → [1, 2, 3, 4, 5] (unchanged)
```

**Pitfall 3: Data type mismatch**
```python
arr = np.array([1, 2, 3])  # dtype: int64
arr = arr / 2  # → [0.5, 1.0, 1.5] (becomes float64)
arr = np.array([1, 2, 3], dtype=np.float32)  # Explicit dtype
```

---

## PART 3: Pandas Core Operations

### Series vs DataFrame

**Series (1D, labeled):**
```python
# From list
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# a    1
# b    2
# c    3
# d    4

s['b']  # → 2
s[['b', 'd']]  # → Series with b, d
```

**DataFrame (2D, labeled rows & columns):**
```python
df = pd.DataFrame({
    'age': [25, 35, 45],
    'income': [30000, 45000, 65000],
    'default': [1, 0, 0]
})
df['age']  # → Series
df.loc[0]  # → Series (first row)
df.iloc[0, 1]  # → 30000 (first row, second column)
```

### Selection & Filtering

**Column selection:**
```python
# Single column (returns Series)
loans['income']

# Multiple columns (returns DataFrame)
loans[['age', 'income']]

# By position
loans.iloc[:, 0:3]  # First 3 columns
```

**Row filtering:**
```python
# Single condition
high_income = loans[loans['income'] > 40000]

# Multiple conditions (use &, |, ~ for AND, OR, NOT)
risky = loans[(loans['income'] < 40000) & (loans['credit_score'] < 700)]
# NOT: loans[loans['income'] < 40000 and loans['credit_score'] < 700]  ❌

# Using isin()
bucket_a = loans[loans['income'].isin([30000, 45000])]
```

**Row selection:**
```python
loans.loc[0]  # First row (by label)
loans.iloc[0]  # First row (by position)
loans.loc[0:2]  # Rows 0-2 (inclusive on both ends!)
loans.iloc[0:2]  # Rows 0-1 (exclusive on right)
```

### GroupBy & Aggregation

**Basic groupby:**
```python
# Group by income bucket, count defaults
loans.groupby('default')['income'].sum()
# default
# 0    415000  (non-defaulters)
# 1     73000  (defaulters)

# Multiple aggregations
loans.groupby('default').agg({
    'income': ['sum', 'mean'],
    'age': 'mean',
    'credit_score': 'max'
})
```

**Advanced: Custom aggregation function**
```python
# Define custom aggregation
def income_to_age_ratio(group):
    return group['income'].sum() / group['age'].mean()

loans.groupby('default').apply(income_to_age_ratio)
```

**Tricky: GroupBy with multiple keys**
```python
# Create income bucket first
loans['income_bucket'] = pd.cut(
    loans['income'], 
    bins=[0, 40000, 60000, float('inf')],
    labels=['low', 'mid', 'high']
)

# Group by multiple keys
result = loans.groupby(['income_bucket', 'default']).size()
# income_bucket  default
# low            0           3
#                1           2
# mid            0           2
#                1           0
# high           0           2
#                1           0
```

**Interview takeaway:**
> "GroupBy + agg is how I think about SQL GROUP BY. If I need custom logic, I use .apply() with a function, but I prefer avoiding it because it's slower than vectorized operations."

### Joins & Merges

**Inner merge (SQL INNER JOIN):**
```python
# Loan data with customer demographics
customers = pd.DataFrame({
    'customer_id': [101, 102, 103],
    'name': ['Alice', 'Bob', 'Charlie']
})

merged = pd.merge(
    loans, customers,
    on='customer_id',
    how='inner'
)
# Only matching customer_ids retained
```

**Left merge (SQL LEFT JOIN):**
```python
merged = pd.merge(
    loans, customers,
    on='customer_id',
    how='left'
)
# All rows from loans, add customer names where available
```

**Outer merge (SQL FULL OUTER JOIN):**
```python
merged = pd.merge(
    loans, customers,
    on='customer_id',
    how='outer'
)
# All rows from both DataFrames
```

**Tricky: Multiple join keys**
```python
orders = pd.DataFrame({
    'customer_id': [101, 102, 101],
    'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'amount': [100, 200, 150]
})

merged = pd.merge(
    loans, orders,
    on='customer_id',
    how='left'
)
# Creates Cartesian product for matching keys (e.g., 101 joins twice)
# Result: loans rows × matching orders rows
```

### Apply, Map, Transform

**apply(): Row or column-wise operation**
```python
# Apply function to each row
def risk_score(row):
    return row['income'] / row['loan_amount']

loans['risk_score'] = loans.apply(risk_score, axis=1)
# axis=0 (default): apply to columns
# axis=1: apply to rows

# Shorthand with lambda
loans['income_k'] = loans['income'].apply(lambda x: x / 1000)
```

**map(): Series value replacement**
```python
# Map categories
loans['default_category'] = loans['default'].map({
    0: 'No Default',
    1: 'Default'
})

# With function
loans['income_category'] = loans['income'].map(
    lambda x: 'high' if x > 50000 else 'low'
)
```

**transform(): Return same shape**
```python
# Standardize income within each default group
loans['income_std'] = loans.groupby('default')['income'].transform(
    lambda x: (x - x.mean()) / x.std()
)
# Returns Series with same index as original
```

**Pitfall: Performance**
```python
# SLOW:
loans['risk'] = loans.apply(lambda row: row['income'] / row['loan_amount'], axis=1)

# FAST (vectorized):
loans['risk'] = loans['income'] / loans['loan_amount']
```

**Interview takeaway:**
> "I avoid .apply() with axis=1 on large DataFrames because it's essentially a Python loop. I vectorize with column operations first, and only use .apply() when truly necessary."

---

## PART 4: Pandas ↔ SQL Mental Model

| SQL | Pandas |
|-----|--------|
| `SELECT col1, col2 FROM table` | `df[['col1', 'col2']]` |
| `WHERE condition` | `df[df['col'] > value]` |
| `GROUP BY col ORDER BY agg DESC` | `df.groupby('col').agg(...).sort_values(ascending=False)` |
| `INNER JOIN table2 ON id` | `pd.merge(df1, df2, on='id', how='inner')` |
| `COUNT(*)` | `len(df)` or `df.shape[0]` |
| `COUNT(DISTINCT col)` | `df['col'].nunique()` |
| `SUM(col)` | `df['col'].sum()` |
| `AVG(col)` | `df['col'].mean()` |
| `MAX(col)` | `df['col'].max()` |
| `CASE WHEN ... THEN ... ELSE ... END` | `np.where()` or `.map()` |
| `RANK() OVER (PARTITION BY col ORDER BY col2)` | `df.groupby('col')['col2'].rank()` |
| `LAG(col) OVER (ORDER BY date)` | `df.sort_values('date')['col'].shift(1)` |
| `ROW_NUMBER() OVER (ORDER BY col)` | `df.sort_values('col').reset_index(drop=True).reset_index()['index'] + 1` |

**Example: SQL to Pandas translation**
```sql
-- SQL
SELECT 
    default, 
    COUNT(*) as count,
    AVG(income) as avg_income
FROM loans
WHERE age > 30
GROUP BY default
ORDER BY count DESC
```

```python
# Pandas
(loans[loans['age'] > 30]
 .groupby('default')
 .agg(count=('income', 'size'), avg_income=('income', 'mean'))
 .sort_values('count', ascending=False)
)
```

---

## PART 5: Tricky Pandas Patterns

### Multi-Index (Hierarchical Index)

**Create multi-index:**
```python
# From groupby result
multi = loans.groupby(['default', 'income_bucket']).size()
# Multi-indexed Series with level 0: default, level 1: income_bucket

# Access specific level
multi.loc[1, :]  # All rows where default=1
multi.loc[(1, 'high')]  # default=1 AND income_bucket='high'
```

**Reset multi-index:**
```python
df = multi.reset_index(name='count')
# Converts back to regular DataFrame
```

### Pivot & Reshape

**Pivot table (like SQL PIVOT):**
```python
# Create cross-tab of default status by income bucket
pivot = loans.pivot_table(
    values='income',
    index='income_bucket',
    columns='default',
    aggfunc='mean'  # or 'sum', 'count'
)
# Rows: income buckets, Columns: default (0, 1)
# Values: average income

# With multiple aggregations
pivot = loans.pivot_table(
    values=['income', 'age'],
    index='income_bucket',
    columns='default',
    aggfunc={'income': 'sum', 'age': 'mean'}
)
```

**Crosstab (categorical cross-tabulation):**
```python
ct = pd.crosstab(
    loans['income_bucket'],
    loans['default'],
    margins=True  # Adds totals
)
```

**Melt (unpivot):**
```python
# Reshape from wide to long
wide = pd.DataFrame({
    'id': [1, 2, 3],
    'q1_score': [90, 85, 88],
    'q2_score': [92, 87, 90]
})

long = wide.melt(
    id_vars='id',
    var_name='quarter',
    value_name='score'
)
# id  quarter  score
# 1   q1_score  90
# 1   q2_score  92
# ...
```

### Window Functions in Pandas

**Rank within group (equivalent to SQL RANK OVER):**
```python
# Rank customers by income within each default group
loans['income_rank'] = loans.groupby('default')['income'].rank(ascending=False)
```

**Cumulative sum:**
```python
# Running total of loan amounts sorted by customer_id
loans = loans.sort_values('customer_id')
loans['cumsum'] = loans.groupby('default')['loan_amount'].cumsum()
```

**Shift (lag/lead):**
```python
# Previous customer's income
loans['prev_income'] = loans.groupby('default')['income'].shift(1)

# Next customer's income
loans['next_income'] = loans.groupby('default')['income'].shift(-1)
```

### Handling Missing Data

**Detect:**
```python
loans.isnull()  # Boolean DataFrame
loans.isnull().sum()  # Count per column
loans.dropna()  # Remove rows with ANY null
loans.dropna(subset=['income'])  # Remove if 'income' is null
loans.fillna(0)  # Replace with 0
loans.fillna(loans['income'].mean())  # Fill with column mean
loans.fillna(method='ffill')  # Forward fill (copy previous value)
```

**Tricky: fillna with groupby**
```python
# Fill missing income with group mean
loans['income'] = loans.groupby('default')['income'].transform(
    lambda x: x.fillna(x.mean())
)
```

---

## PART 6: File I/O & String Tricks

### Reading Files (Performance Matters)

**CSV reading with chunks (large files):**
```python
# Read 10k rows at a time
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)  # Process each chunk
    # Avoids loading entire file into memory
```

**CSV reading with dtype specification:**
```python
# Avoid type inference on large files (slow)
df = pd.read_csv(
    'data.csv',
    dtype={'customer_id': 'int32', 'income': 'float32', 'default': 'int8'}
)
# Reduces memory by 50%+ if dtypes chosen carefully
```

**Reading from multiple formats:**
```python
pd.read_csv('file.csv')
pd.read_parquet('file.parquet')  # Faster, compressed
pd.read_json('file.json')
pd.read_sql_query('SELECT * FROM table', connection)  # From database
```

**Writing efficiently:**
```python
df.to_csv('output.csv', index=False)
df.to_parquet('output.parquet')  # Better for repeated reads
df.to_sql('table_name', connection, if_exists='replace')
```

### String Operations

**String methods on Series:**
```python
names = pd.Series(['alice', 'bob', 'charlie'])

names.str.upper()  # → ['ALICE', 'BOB', 'CHARLIE']
names.str.len()  # → [5, 3, 7]
names.str.contains('li')  # → [True, False, False]
names.str.split('c')  # → Split by character
names.str.replace('a', 'X')  # → 'Xlice', 'bob', 'chXrlie'
```

**Extract substrings:**
```python
ids = pd.Series(['2024_abc_100', '2024_def_200'])
extracted = ids.str.extract(r'(\d{4})_(\w+)_(\d+)')
# Column 0: year, Column 1: code, Column 2: amount
```

---

## PART 7: Performance & Interview Tricks

### Speed Checklist

```python
# SLOW (avoid):
for i, row in df.iterrows():
    process(row)

# FAST (prefer):
df.apply(process, axis=1)  # Still slow but faster than iterrows

# FASTEST:
df['result'] = df['col1'] + df['col2']  # Vectorized
```

### Memory Optimization

```python
# Default dtypes are memory-heavy
df['age'] = df['age'].astype('int8')  # int64 → int8 (saves 87.5%)
df['score'] = df['score'].astype('float32')  # float64 → float32 (saves 50%)

# Categorical for many repeated values
df['category'] = df['category'].astype('category')  # Stores unique values once

# Check memory
df.memory_usage(deep=True)
```

### Chaining Operations

**Readable and efficient:**
```python
result = (loans
    [loans['age'] > 30]
    .groupby('income_bucket')
    .agg({'income': 'sum', 'age': 'mean'})
    .sort_values('income', ascending=False)
    .reset_index()
)
```

---

## PART 7B: Programming Paradigms & Python's Ecosystem

### Functional Programming (FP)

**What it is:** Programming with functions as first-class objects; avoid mutable state; emphasis on composability.

**Core idea:** Instead of objects with state, pass data through pure functions.

**Python's FP tools:**

1. **map()** — Apply function to each element
```python
scores = [650, 720, 780]
scaled = list(map(lambda x: x / 100, scores))
# → [6.5, 7.2, 7.8]

# Pandas equivalent (faster):
scores_series = pd.Series(scores)
scaled = scores_series / 100
```

2. **filter()** — Keep elements matching condition
```python
scores = [650, 720, 780, 600]
high_scores = list(filter(lambda x: x > 700, scores))
# → [720, 780]

# Pandas equivalent:
high_scores = scores_series[scores_series > 700].tolist()
```

3. **reduce()** — Accumulate into single value
```python
from functools import reduce
scores = [650, 720, 780]
product = reduce(lambda x, y: x * y, scores)
# → 650 * 720 * 780 = 364,800,000
```

**Why FP matters in data science:**
- **Composability:** Chain transformations (map → filter → reduce) without intermediate state
- **Testability:** Pure functions (same input → same output) are easier to test
- **Parallelization:** Stateless functions can run in parallel safely

**FP in practice (Pandas):**
```python
# Functional style chain
result = (loans
    .assign(income_scaled=lambda df: df['income'] / 1000)  # map
    .query('income_scaled > 40')  # filter
    .groupby('default')['income_scaled'].sum()  # reduce
)
```

**Interview takeaway:**
> "Functional programming emphasizes immutability and pure functions. In Python, I use it for data transformation pipelines—chaining operations without side effects makes debugging easier. Pandas methods like `.assign()` and `.query()` support this style naturally."

---

### Object-Oriented Programming (OOP)

**What it is:** Organize code around objects (data + methods); encourage reusability and encapsulation.

**Core concepts:**

1. **Classes and Attributes**
```python
class Loan:
    def __init__(self, customer_id, income, loan_amount):
        self.customer_id = customer_id
        self.income = income
        self.loan_amount = loan_amount
    
    def risk_ratio(self):
        return self.loan_amount / self.income

# Usage
loan = Loan(101, 50000, 10000)
print(loan.risk_ratio())  # → 0.2
```

2. **Inheritance** — Reuse parent class behavior
```python
class RiskyLoan(Loan):
    def __init__(self, customer_id, income, loan_amount, interest_rate):
        super().__init__(customer_id, income, loan_amount)
        self.interest_rate = interest_rate
    
    def total_cost(self):
        return self.loan_amount * (1 + self.interest_rate)

risky = RiskyLoan(101, 50000, 10000, 0.05)
print(risky.total_cost())  # → 10500
```

3. **Encapsulation** — Hide internal state
```python
class Account:
    def __init__(self, balance):
        self._balance = balance  # Convention: leading _ means "private"
    
    def deposit(self, amount):
        self._balance += amount
        return self._balance
    
    @property
    def balance(self):
        return self._balance
    
    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = value

account = Account(1000)
account.balance = 500  # Uses setter, validates
print(account.balance)  # → 500
```

4. **Dataclass** — Lightweight class for data
```python
from dataclasses import dataclass

@dataclass
class Customer:
    customer_id: int
    age: int
    income: float
    default: int
    
    def is_high_risk(self):
        return self.income < 40000 and self.default == 1

# Auto-generates __init__, __repr__, __eq__
cust = Customer(101, 25, 30000, 1)
print(cust)  # → Customer(customer_id=101, age=25, ...)
```

**Why OOP matters in data engineering:**
- **Reusability:** Base classes for data pipelines, validators, transformers
- **Encapsulation:** Hide complexity (e.g., database connection logic)
- **Testing:** Mock objects for unit testing

**OOP in pipelines (example):**
```python
class DataTransformer:
    """Base class for all transformations"""
    def transform(self, df):
        raise NotImplementedError

class NormalizeIncomes(DataTransformer):
    def transform(self, df):
        df['income_normalized'] = df['income'] / df['income'].max()
        return df

class FlagDefaults(DataTransformer):
    def transform(self, df):
        df['is_default'] = df['default'].astype(bool)
        return df

# Chain transformations
transformers = [NormalizeIncomes(), FlagDefaults()]
result = loans
for transformer in transformers:
    result = transformer.transform(result)
```

**Interview takeaway:**
> "I use OOP to structure data pipelines—base classes for common patterns, subclasses for specific logic. This avoids code duplication and makes testing easier. For simple data transformations, dataclasses are cleaner than writing full classes."

---

### Dynamic Programming (DP)

**What it is:** Solve complex problems by breaking them into overlapping subproblems; cache intermediate results to avoid recomputation.

**Key insight:** If you see a problem that could be solved recursively but with repeated calculations, use DP.

**Classic example: Fibonacci**
```python
# ❌ Naive recursion (exponential time)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
# fib(5) recalculates fib(3) many times

# ✅ DP with memoization (linear time)
def fib_memo(n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fib_memo(n-1, cache) + fib_memo(n-2, cache)
    return cache[n]

# Or bottom-up DP (no recursion)
def fib_dp(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

**Data engineering relevance (rare, but can appear):**

Example: **Longest increasing subsequence in time-series data**
```python
# Given a stream of click counts over time, find longest streak of increasing counts
def lis_length(arr):
    """Longest increasing subsequence length"""
    n = len(arr)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0

clicks = [3, 1, 4, 1, 5, 9, 2, 6]
print(lis_length(clicks))  # → 5 (e.g., [1, 4, 5, 9])
```

**When DP appears in data eng interviews:**
- **Cost optimization:** "Find cheapest path through a pipeline of choices"
- **Time-series patterns:** "Longest increasing/decreasing trend"
- **Resource allocation:** "Optimal split of compute budget"

**Interview takeaway:**
> "Dynamic programming is useful when you have overlapping subproblems. In data engineering, it's less common than in pure algorithms, but it can show up in optimization problems (e.g., pipeline scheduling). I'd recognize it as a DP problem, think about state transitions, and implement memoization to avoid redundant calculations."

---

### Why Python Dominates Data Science & ML

#### 1. **Rich Ecosystem (NumPy, Pandas, Scikit-learn)**

**The NumPy foundation:**
- C-compiled arrays → 100–1000x faster than Python loops
- Linear algebra, random number generation, Fourier transforms all built-in
- Every data science library builds on NumPy

```python
# This is the killer: NumPy operations are fast
a = np.array([1, 2, 3]) * 1000000
# Parallelized, compiled C code—not Python loops
```

**Pandas: SQL + Excel in Python**
- Most data scientists come from SQL/Excel backgrounds
- Pandas syntax maps directly to SQL operations (GROUP BY → groupby)
- Exploratory analysis is 10x faster than writing SQL + exporting

**Scikit-learn: Unified API**
```python
# All ML algorithms follow same interface
clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# Switch to random forest, no syntax change
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

#### 2. **Low Barrier to Entry**

**Readable syntax:**
```python
# Python
for customer_id, income in zip(ids, incomes):
    if income > 50000:
        print(f"High earner: {customer_id}")

# Java equivalent would need 5x more lines
```

**Interactive notebooks (Jupyter):**
- Experiment iteratively without recompiling
- Mix code + markdown + visualizations
- Standard in data science (almost unthinkable without it now)

#### 3. **Bridging Worlds: Research → Production**

**Data scientist can prototype quickly:**
```python
# Day 1: Experiment in Jupyter
model = XGBClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

**Same engineer can deploy to production:**
```python
# Week 2: Wrap in Flask + Docker
from flask import Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pred = model.predict([data['features']])
    return jsonify({'prediction': pred[0]})
```

No context switching between languages (R → Java, MATLAB → C++).

#### 4. **Strong Statistical & ML Libraries**

| Task | Library | Why Python Won |
|------|---------|-----------------|
| NumPy math | NumPy | Vectorized, fast |
| Data manipulation | Pandas | SQL-like, intuitive |
| ML models | Scikit-learn | Unified API, documentation |
| Deep learning | PyTorch, TensorFlow | GPU support, research-friendly |
| Visualization | Matplotlib, Seaborn | Quick EDA plots |
| Statistical tests | SciPy | Comprehensive |
| Time-series | Statsmodels | ARIMA, forecasting built-in |

#### 5. **Community & Momentum**

- Kaggle competitions → Python (not R, not Scala)
- Research papers include Python code → reproducibility
- GitHub + Stack Overflow → massive knowledge base
- Attracts talent cycle: more DS join → more libraries → attracts more DS

#### 6. **Flexibility (Rapid Iteration)**

```python
# Quick experiment 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Quick experiment 2 (different split)
X_train, X_test = cross_val_split(X, y, n_splits=5)

# No recompile, no restart
```

Compare to Java/C++: change anything → recompile → restart → minutes of overhead.

#### 7. **Integration with Systems (Your Advantage)**

At Adform, you probably use:
```python
# Vertica + Python
import pyodbc
conn = pyodbc.connect('Driver=Vertica...')
df = pd.read_sql('SELECT * FROM bidding_data', conn)

# Spark + Python (Pyspark)
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('RTB').getOrCreate()
df = spark.sql('SELECT * FROM events')

# Kafka + Python (streaming)
from kafka import KafkaConsumer
for msg in consumer:
    process(msg)
```

One language connects to databases, big data systems, message queues.

---

#### Why Not Other Languages?

**R:** Great statistics, poor software engineering. Hard to scale to production.

**Scala/Java:** Production-ready, but steep learning curve. Scientists avoid it.

**Rust:** Fast, safe, but too complex for rapid experimentation.

**Go:** Good systems programming, bad for numerical computing.

**Julia:** Technically superior for math, but tiny ecosystem compared to Python.

---

### Interview Angle: Why You Use Python

**If asked "Why Python for data engineering?":**

> "Python bridges exploration and production. I can experiment in Jupyter, validate on a sample, then scale the same logic to Spark or database. The NumPy/Pandas foundation is rock-solid and fast. And because every data scientist and engineer knows Python, it's the lingua franca—integrating with Vertica, Spark, Airflow all becomes straightforward."

**If asked "Python vs Scala for Spark?":**

> "I'd use Python/Pyspark for data transformation and feature engineering because I think in Pandas/SQL. For jobs where raw speed matters (XGBoost training at scale, heavy numerical computation), Scala Spark might be faster, but Python's overhead is small compared to I/O. At Adform, I use Python because the team standardized on it and it's fast enough."

**If asked "When would you NOT use Python?":**

> "High-frequency trading or microsecond-latency systems where Python's GIL (Global Interpreter Lock) becomes a bottleneck. Or if the system required extreme type safety at compile time. For most data work, Python's flexibility and speed of development outweighs the performance cost."

---

## PART 8: Interview Talking Points

### When asked "How would you process a large CSV?"

> "I'd use `pd.read_csv()` with `chunksize` and `dtype` specified to avoid loading everything into memory and to skip type inference. For repeated reads, I'd convert to Parquet. If it's really large (>100GB), I'd move to Spark."

### When asked "Difference between .apply() and vectorized operations?"

> ".apply() iterates row-by-row in Python, which is slow. Vectorized operations (like `df['a'] + df['b']`) use compiled NumPy code underneath. On 1M rows, vectorized is 100–1000x faster. I only use .apply() when there's no vectorized alternative."

### When asked "How do you handle missing data?"

> "It depends on context. If the missingness is random and <5%, I'd drop it. If it's systematic (e.g., new customers with no history), I'd fill with group mean or a sensible default. I'd flag missing data in logs to catch upstream issues early."

### When asked "Pandas vs SQL for aggregations?"

> "SQL is faster for large datasets because it runs on the database where the data lives. Pandas is better for exploratory analysis and when I need to pivot/reshape. For production pipelines, I'd do heavy lifting in SQL and use Pandas for final transformations in Python."

### When asked about joins with duplicate keys

> "When joining on a key with multiple matches, Pandas creates a Cartesian product. If customer_id=101 appears 2 times in loans and 3 times in orders, the merged result has 6 rows. I always verify the shape before and after to catch unintended duplicates."

---

## Quick Reference Cheat Sheet

### Pandas
- `df.shape` — (rows, columns)
- `df.info()` — Data types and nulls
- `df.describe()` — Summary stats
- `df.value_counts()` — Frequency table
- `df.duplicated()` — Find duplicates
- `df.drop_duplicates()` — Remove duplicates
- `df.sort_values('col')` — Sort
- `df.sample(n=10)` — Random sample

### NumPy
- `np.unique(arr)` — Unique values
- `np.where(condition, ifTrue, ifFalse)` — Conditional replacement
- `np.concatenate([arr1, arr2])` — Append arrays
- `np.dot(a, b)` — Matrix multiplication
- `np.random.seed(42)` — Reproducibility

### Comprehensions
- List: `[expr for x in iter if cond]`
- Dict: `{k: v for x in iter if cond}`
- Generator: `(expr for x in iter if cond)`

---

## Practice Challenges

**Challenge 1: Feature engineering**
```python
# Given loans DataFrame, create:
# 1. debt_to_income = loan_amount / income
# 2. age_bucket (young: <30, mid: 30-40, senior: >40)
# 3. avg_income_by_bucket (apply to each row)
```

**Challenge 2: Window aggregation**
```python
# For each customer's default status, rank by income (descending)
# Then get the top 3 incomes per default group
```

**Challenge 3: Merge and reshape**
```python
# Merge houses and loans on a made-up key
# Pivot to show avg house price by bedrooms and age_bucket
```

**Challenge 4: String operations**
```python
# From a list of emails like 'alice@company.com', extract username
# Count how many times each domain appears
```

**Challenge 5: Groupby with custom function**
```python
# Group loans by default, then compute:
# - coefficient of variation (std / mean) of income within each group
# - percentage of total loan amount
```

---

## Final Takeaway

**Your interview story:**
> "I think of Pandas as the Python interface to SQL operations. I'm comfortable with vectorized NumPy operations because they're fast, and I understand the memory/speed tradeoffs (e.g., int32 vs int64). On large data, I transition to Spark or SQL early. I avoid .apply() row-wise loops and prefer groupby + agg patterns that map cleanly to SQL thinking."
