# SQL Interview Questions - MySQL Solutions

A comprehensive collection of SQL interview questions with detailed explanations and MySQL solutions.

---

## Table of Contents

1. [Repeated Payments](#question-1-repeated-payments) - Stripe
2. [Median Google Search Frequency](#question-2-median-google-search-frequency) - Google
3. [Monthly Merchant Balance](#question-3-monthly-merchant-balance) - Visa
4. [Concept: Correlated Subqueries](#concept-correlated-subqueries)
5. [Server Utilization Time](#question-4-server-utilization-time) - Amazon
6. [Uniquely Staffed Consultants](#question-5-uniquely-staffed-consultants) - Accenture
7. [Event Friends Recommendation](#question-6-event-friends-recommendation) - Facebook
8. [3-Topping Pizzas](#question-7-3-topping-pizzas) - McKinsey
9. [Follow-Up AirPods Percentage](#question-8-follow-up-airpods-percentage) - Apple

---

## Question 1: Repeated Payments

**Source:** Stripe SQL Interview Question

---

### Problem Description

Identify any payments made at the same merchant with the same credit card for the same amount within 10 minutes of each other and report the count of such repeated payments.

This is a classic fraud detection scenario where duplicate transactions suggest potential malicious activity or system errors.

---

### Example Input

transactions table:

| transaction_id | merchant_id | credit_card_id | amount | transaction_timestamp |
|---|---|---|---|---|
| 1 | 101 | 1 | 100 | 2022-09-25 12:00:00 |
| 2 | 101 | 1 | 100 | 2022-09-25 12:08:00 |
| 3 | 101 | 1 | 100 | 2022-09-25 12:28:00 |
| 4 | 102 | 2 | 300 | 2022-09-25 12:00:00 |
| 6 | 102 | 2 | 400 | 2022-09-25 14:00:00 |

---

### Example Output

| payment_count |
|---|
| 1 |

**Explanation:**
- Transactions 1 & 2: Same merchant (101), same card (1), same amount (100), time difference = 8 minutes ✓ (within 10 minutes)
- Transactions 2 & 3: Same merchant (101), same card (1), same amount (100), time difference = 20 minutes ✗ (exceeds 10 minutes)
- Transactions 4 & 6: Different amounts (300 vs 400) ✗ (not repeated)

Result: 1 repeated payment detected (the 8-minute gap between transactions 1 & 2)

---

### MySQL Solution

```sql
WITH payments AS (
  SELECT 
    merchant_id,
    TIMESTAMPDIFF(MINUTE, 
      LAG(transaction_timestamp) OVER(
        PARTITION BY merchant_id, credit_card_id, amount
        ORDER BY transaction_timestamp
      ),
      transaction_timestamp
    ) AS minute_difference
  FROM transactions
)
SELECT COUNT(merchant_id) AS payment_count 
FROM payments 
WHERE minute_difference <= 10;
```

---

### Query Explanation

**Step 1:** `LAG()` with `PARTITION BY merchant_id, credit_card_id, amount` gets previous transaction's timestamp for identical transactions.

**Step 2:** `TIMESTAMPDIFF(MINUTE, ...)` calculates time gap between consecutive identical transactions.

**Step 3:** Count rows where gap ≤ 10 minutes.

---

### Key Concepts

- **Window Functions:** `LAG()` to access previous transaction
- **Time Calculation:** `TIMESTAMPDIFF()` for minute differences
- **Partitioning:** Groups identical merchant/card/amount combinations

---

## Question 2: Median Google Search Frequency

**Source:** Google SQL Interview Question

---

### Problem Description

Google's Marketing Team needs to calculate a statistic for their Superbowl Ad: the median number of searches made per user per year. Given a summary table that aggregates search data, write a query to report the median searches made per user.

This requires expanding aggregated data back to individual records and computing the median value.

---

### Example Input

search_frequency table:

| searches | num_users |
|---|---|
| 1 | 2 |
| 2 | 2 |
| 3 | 3 |
| 4 | 1 |

---

### Example Output

| median |
|---|
| 2.5 |

**Explanation:**
Expanding the data: `[1, 1, 2, 2, 3, 3, 3, 4]` (8 total values)
- Sorted position: 1st, 2nd, 3rd, 4th, 5th, 6th, 7th, 8th
- Median is average of 4th and 5th values: (2 + 3) / 2 = **2.5**

---

### MySQL Solution

```sql
WITH RECURSIVE cte_nums AS (
  SELECT 1 as n, (SELECT MAX(num_users) FROM search_frequency) as max_n
  UNION ALL
  SELECT n + 1, max_n FROM cte_nums WHERE n < max_n
),
searches_expanded AS (
  SELECT sf.searches
  FROM search_frequency sf
  CROSS JOIN cte_nums
  WHERE cte_nums.n <= sf.num_users
),
ranked_searches AS (
  SELECT 
    searches,
    ROW_NUMBER() OVER (ORDER BY searches) as row_num,
    COUNT(*) OVER () as total_count
  FROM searches_expanded
)
SELECT 
  ROUND(AVG(searches), 1) as median
FROM ranked_searches
WHERE row_num IN (
  FLOOR((total_count + 1) / 2),
  CEIL((total_count + 1) / 2)
);
```

---

### Query Explanation

**The Core Challenge:** Input is compressed (aggregated). Median needs individual observations.

**Solution Pattern:**
```
Compressed data → Expand to individual rows → Sort → Find middle value(s)
```

---

## Approach A: Recursive CTE (Direct Expansion)

**Intuition: "Generate Rows by Counting Down"**

The core idea: For each (searches, num_users) row, we want to create `num_users` copies of the `searches` value. 

A recursive CTE does this by starting with `num_users` and counting DOWN to 0, producing one output row per iteration.

---

### The Pattern: Countdown Counter

Think of it like a production line:
- **Input:** searches=3, num_users=3 (make 3 copies of value 3)
- **Counter starts at:** 3
- **Each step:** Produce one row, decrement counter
- **Stop when:** Counter reaches 0

| Step | Counter | Action | Output |
|---|---|---|---|
| 1 | 3 | Produce row | `3` |
| 2 | 2 | Produce row | `3` |
| 3 | 1 | Produce row | `3` |
| STOP | 0 | Counter is 0, stop | — |

Result: 3 identical rows ✓

---

### How the SQL Code Works

Here's the actual recursive CTE pattern:

```sql
WITH RECURSIVE countdown AS (
  SELECT searches, num_users - 1 AS counter
  FROM search_frequency
  
  UNION ALL
  
  SELECT searches, counter - 1
  FROM countdown
  WHERE counter > 0  ← Stop when counter reaches 0
)
SELECT searches FROM countdown;
```

**Two Parts:**

1. **ANCHOR (Base Case):** Start with `num_users - 1` as the counter
2. **RECURSIVE Part:** Keep subtracting 1 from counter UNTIL it becomes 0

Each iteration produces ONE row with the `searches` value.

---

### Step-by-Step with Your Actual Data

Start with:
| searches | num_users |
|---|---|
| 1 | 2 |
| 2 | 2 |
| 3 | 3 |
| 4 | 1 |

**For searches=1, num_users=2:**

| Iteration | counter value | Output row | Continue? |
|---|---|---|---|
| 1 (anchor) | 1 | `1` | Yes (1 > 0) |
| 2 (recursive) | 0 | `1` | No (0 > 0 is FALSE) |

Result: 2 rows of value `1` ✓

**For searches=3, num_users=3:**

| Iteration | counter value | Output row | Continue? |
|---|---|---|---|
| 1 (anchor) | 2 | `3` | Yes (2 > 0) |
| 2 (recursive) | 1 | `3` | Yes (1 > 0) |
| 3 (recursive) | 0 | `3` | No (0 > 0 is FALSE) |

Result: 3 rows of value `3` ✓

---

### Full Expansion Result

All iterations across all input rows produce:

| searches |
|---|
| 1 |
| 1 |
| 2 |
| 2 |
| 3 |
| 3 |
| 3 |
| 4 |

Each value appears exactly as many times as its `num_users`. ✓

---

### Then Rank and Find Median

| searches | row_num |
|---|---|
| 1 | 1 |
| 1 | 2 |
| 2 | 3 |
| 2 | 4 |
| 3 | 5 |
| 3 | 6 |
| 3 | 7 |
| 4 | 8 |

For 8 rows (even):
- Middle positions: `FLOOR((8+1)/2) = 4` and `CEIL((8+1)/2) = 5`
- Values at those positions: **2** and **3**
- Median = (2 + 3) / 2 = **2.5** ✓

---

## Approach B: Generate Numbers + CROSS JOIN (Alternative)

**Step 1: Generate a Numbers Table**

Create numbers 1 through MAX(num_users):

| n |
|---|
| 1 |
| 2 |
| 3 |

**Step 2: CROSS JOIN (Every Combination)**

Combine numbers with your frequency table:

```
search_frequency × nums (every pairing)
```

| searches | num_users | n |
|---|---|---|
| 1 | 2 | 1 |
| 1 | 2 | 2 |
| 1 | 2 | 3 |
| 2 | 2 | 1 |
| 2 | 2 | 2 |
| 2 | 2 | 3 |
| 3 | 3 | 1 |
| 3 | 3 | 2 |
| 3 | 3 | 3 |
| 4 | 1 | 1 |
| 4 | 1 | 2 |
| 4 | 1 | 3 |

**Step 3: Filter WHERE n ≤ num_users**

Keep only rows where the number is ≤ the count:

| searches |
|---|
| 1 |
| 1 |
| 2 |
| 2 |
| 3 |
| 3 |
| 3 |
| 4 |

Same result! Then rank and calculate median.

---

**Why Two Approaches?**

| Aspect | Approach A (Recursive) | Approach B (CROSS JOIN) |
|---|---|---|
| **Intuition** | "Keep subtracting until done" | "Generate all numbers, filter" |
| **Conceptual** | Direct countdown | Parallels PostgreSQL `GENERATE_SERIES()` |
| **Best for** | Understanding recursion | Scaling to large `num_users` |

---

### Key Concepts

- **Recursive CTE:** Countdown counter produces one row per frequency count
- **CROSS JOIN:** Creates every pairing; filter to select relevant ones
- **De-aggregation:** Expands summary data into individual observations
- **Median Calculation:** `ROW_NUMBER()` + middle position logic works for odd/even

---

## Question 3: Monthly Merchant Balance

**Source:** Visa SQL Interview Question

---

### Problem Description

Given a transaction table with deposits and withdrawals, calculate the cumulative balance of a merchant account at the end of each day. The cumulative balance resets to zero at the end of each month (monthly partition).

This is a classic problem combining transaction sign logic (deposits +, withdrawals −) with running totals and partition resets.

---

### Example Input

transactions table:

| transaction_id | type | amount | transaction_date |
|---|---|---|---|
| 19153 | deposit | 65.90 | 2022-07-10 10:00:00 |
| 53151 | deposit | 178.55 | 2022-07-08 10:00:00 |
| 29776 | withdrawal | 25.90 | 2022-07-08 10:00:00 |
| 16461 | withdrawal | 45.99 | 2022-07-08 10:00:00 |
| 77134 | deposit | 32.60 | 2022-07-10 10:00:00 |

---

### Example Output

| transaction_date | balance |
|---|---|
| 2022-07-08 | 106.66 |
| 2022-07-10 | 205.16 |

**Explanation:**

**July 8th:**
- 53151: +178.55 (deposit)
- 29776: −25.90 (withdrawal)
- 16461: −45.99 (withdrawal)
- **Daily total: 178.55 − 25.90 − 45.99 = 106.66**
- Cumulative: 106.66

**July 10th:**
- 19153: +65.90 (deposit)
- 77134: +32.60 (deposit)
- **Daily total: 65.90 + 32.60 = 98.50**
- Cumulative: 106.66 + 98.50 = **205.16**

---

### MySQL Solution

```sql
WITH daily_balances AS (
  SELECT
    DATE(transaction_date) AS transaction_day,
    DATE_FORMAT(transaction_date, '%Y-%m-01') AS transaction_month,
    SUM(CASE 
      WHEN type = 'deposit' THEN amount
      WHEN type = 'withdrawal' THEN -amount 
    END) AS daily_balance
  FROM transactions
  GROUP BY 
    DATE(transaction_date),
    DATE_FORMAT(transaction_date, '%Y-%m-01')
)
SELECT
  transaction_day,
  SUM(daily_balance) OVER (
    PARTITION BY transaction_month
    ORDER BY transaction_day
  ) AS balance
FROM daily_balances
ORDER BY transaction_day;
```

---

### Query Explanation

**CTE (daily_balances):**
- Group by day and month
- Apply sign logic: `CASE WHEN type='deposit' THEN + ELSE -`
- Result: daily net balance per day

**Main Query:**
- `SUM() OVER (PARTITION BY transaction_month ORDER BY transaction_day)` 
  - Cumulative sum within each month
  - `PARTITION BY month` resets at month boundaries
  - `ORDER BY day` ensures chronological running total

---

### Key Concepts

- **Sign Logic:** `CASE` statement to apply +/− based on transaction type
- **Date Grouping:** Aggregate to daily level before windowing
- **Partition Reset:** `PARTITION BY` ensures cumulative sum resets each month
- **Window Function:** `SUM() OVER (... ORDER BY ...)` creates running total within partition

---

## Concept: Correlated Subqueries

---

### What Are Correlated Subqueries?

A correlated subquery is a subquery that **references columns from the outer query**. Importantly, it executes **once for each row** of the outer query, making it slower than joins but useful for specific scenarios.

---

### Structure

```sql
SELECT column1
FROM table1 AS outer_table
WHERE column2 > (
  SELECT AVG(column2)
  FROM table2
  WHERE table2.id = outer_table.id  ← References outer_table
);
```

---

### Example: Find Merchants Above Average Daily Balance

Given a transactions table, find all merchants whose transaction amount exceeds their personal average:

```sql
SELECT 
  merchant_id,
  amount,
  transaction_date
FROM transactions t1
WHERE amount > (
  SELECT AVG(amount)
  FROM transactions t2
  WHERE t2.merchant_id = t1.merchant_id  ← Correlated condition
);
```

---

### How It Works

For each row in the outer query:
1. Extract the merchant_id (e.g., 101)
2. Execute the inner query: `AVG(amount) WHERE merchant_id = 101`
3. Compare: Is this row's amount > that average?
4. If yes, include the row

---

### When to Use

- ✓ Comparisons to row-specific aggregates (above average, recent, etc.)
- ✗ Better alternatives often exist (window functions, joins)

---

### Modern Alternative (Window Functions)

Correlated subqueries often have faster equivalents using window functions:

```sql
SELECT 
  merchant_id,
  amount,
  transaction_date,
  ROW_NUMBER() OVER (PARTITION BY merchant_id ORDER BY amount DESC) AS rank_in_merchant
FROM transactions
WHERE amount > (
  SELECT AVG(amount)
  FROM transactions
);
```

---

## Question 4: Server Utilization Time

**Source:** Amazon SQL Interview Question

---

### Problem Description

AWS manages a large fleet of servers. To optimize server usage, calculate the total time that the fleet of servers was running. Each server may start and stop multiple times, so sum up each server's individual uptime to get the fleet-wide total.

Output the result in full days.

---

### Example Input

server_utilization table:

| server_id | status_time | session_status |
|---|---|---|
| 1 | 2022-08-02 10:00:00 | start |
| 1 | 2022-08-04 10:00:00 | stop |
| 2 | 2022-08-17 10:00:00 | start |
| 2 | 2022-08-24 10:00:00 | stop |

---

### Example Output

| total_uptime_days |
|---|
| 21 |

**Explanation:**

Server 1 uptime:
- Start: 2022-08-02 10:00:00
- Stop: 2022-08-04 10:00:00
- Duration: 2 days

Server 2 uptime:
- Start: 2022-08-17 10:00:00
- Stop: 2022-08-24 10:00:00
- Duration: 7 days

**Total fleet uptime: 2 + 7 = 9 days**

*(Note: If the expected output is 21 days with your data, verify the full dataset—more start/stop pairs may be present.)*

---

### MySQL Solution

```sql
WITH running_time AS (
  SELECT
    server_id,
    session_status,
    status_time AS start_time,
    LEAD(status_time) OVER (
      PARTITION BY server_id
      ORDER BY status_time
    ) AS stop_time
  FROM server_utilization
)
SELECT
  FLOOR(SUM(TIMESTAMPDIFF(DAY, start_time, stop_time)) / 1.0) AS total_uptime_days
FROM running_time
WHERE session_status = 'start'
  AND stop_time IS NOT NULL;
```

---

### Query Explanation

**CTE (running_time):**
- `LEAD(status_time)` gets next row's timestamp within each server's partition
- Creates pairs: start_time (current row), stop_time (next row)
- 'start' rows get paired with 'stop'; 'stop' rows get NULL

**Main Query:**
- Filter: `WHERE session_status = 'start' AND stop_time IS NOT NULL`
- Calculates: `TIMESTAMPDIFF(DAY, start_time, stop_time)` per pair
- Sums durations across all servers

---

### Key Concepts

- **LEAD():** Window function to retrieve the next row's value within a partition
- **Pairing:** Use window functions to link related start/stop records
- **Partition Logic:** `PARTITION BY server_id` ensures each server's events are grouped separately
- **Filtering:** `WHERE session_status = 'start'` prevents double-counting (only count start rows with valid stop times)
- **Time Arithmetic:** `TIMESTAMPDIFF(DAY, ...)` calculates duration in days

---

## Question 5: Uniquely Staffed Consultants

**Source:** Accenture SQL Interview Question

---

### Problem Description

As a Data Analyst on the People Operations team, analyze consultant staffing across clients. For each client, report:
1. **Total staffed:** How many consultants are assigned to that client (across all engagements)
2. **Exclusive staffed:** How many consultants work for **only that client** (not shared with others)

This requires distinguishing between consultants who are dedicated to a single client vs. those working on multiple clients.

---

### Example Input

employees table:

| employee_id | engagement_id |
|---|---|
| 1001 | 1 |
| 1001 | 2 |
| 1002 | 1 |
| 1003 | 3 |
| 1004 | 4 |

consulting_engagements table:

| engagement_id | project_name | client_name |
|---|---|---|
| 1 | SAP Logistics Modernization | Department of Defense |
| 2 | Oracle Cloud Migration | Department of Education |
| 3 | Trust & Safety Operations | Google |
| 4 | SAP IoT Cloud Integration | Google |

---

### Example Output

| client_name | total_staffed | exclusive_staffed |
|---|---|---|
| Department of Defense | 2 | 1 |
| Department of Education | 1 | 0 |
| Google | 2 | 2 |

**Explanation:**

**Department of Defense (Engagement 1):**
- Employees: 1001, 1002
- Employee 1001: Also works on Engagement 2 (different client) → NOT exclusive
- Employee 1002: Only works on Engagement 1 → EXCLUSIVE
- Total: 2, Exclusive: 1 ✓

**Department of Education (Engagement 2):**
- Employees: 1001
- Employee 1001: Also works on Engagement 1 (different client) → NOT exclusive
- Total: 1, Exclusive: 0 ✓

**Google (Engagements 3, 4):**
- Employees: 1003, 1004
- Employee 1003: Only works on Engagement 3 (same client Google) → EXCLUSIVE
- Employee 1004: Only works on Engagement 4 (same client Google) → EXCLUSIVE
- Total: 2, Exclusive: 2 ✓

---

### MySQL Solution

```sql
WITH exclusive_employees AS (
  SELECT employee_id
  FROM employees
  JOIN consulting_engagements AS ce 
    ON employees.engagement_id = ce.engagement_id
  GROUP BY employee_id
  HAVING COUNT(DISTINCT ce.client_name) = 1
)
SELECT 
  ce.client_name, 
  COUNT(DISTINCT employees.employee_id) AS total_staffed,
  COUNT(DISTINCT ee.employee_id) AS exclusive_staffed
FROM employees
INNER JOIN consulting_engagements AS ce 
  ON employees.engagement_id = ce.engagement_id
LEFT JOIN exclusive_employees AS ee 
  ON employees.employee_id = ee.employee_id
GROUP BY ce.client_name
ORDER BY ce.client_name;
```

---

### Query Explanation

**CTE (exclusive_employees):**
- Join employees with consulting_engagements
- `HAVING COUNT(DISTINCT client_name) = 1` filters to consultants working for only 1 client
- Result: List of exclusive employee IDs

**Main Query:**
- INNER JOIN employees to consulting_engagements for all staffed count
- LEFT JOIN exclusive_employees to identify exclusive rows
- GROUP BY client_name and COUNT both total and exclusive employee IDs

---

### Key Concepts

- **Distinct Counting:** `COUNT(DISTINCT client_name)` prevents double-counting when an employee has multiple engagements with the same client
- **Two-Level Grouping:** First identify exclusive employees, then count by client
- **LEFT JOIN vs INNER JOIN:** INNER JOIN counts all staffed; LEFT JOIN + counting non-NULL shows exclusive
- **HAVING Clause:** Filters groups (employees) based on aggregate conditions
- **Multiple Joins:** Combines employee assignments with client info, then cross-references the exclusive list

---

## Question 6: Event Friends Recommendation

**Source:** Facebook SQL Interview Question

---

### Problem Description

Facebook wants to recommend new friends by identifying people who show interest in attending 2 or more of the same private events but are not yet friends.

This requires finding: (1) shared private event attendance, (2) current non-friendship status, and (3) pairs with sufficient overlap.

---

### Example Input

friendship_status table:

| user_a_id | user_b_id | status |
|---|---|---|
| 111 | 333 | not_friends |
| 222 | 333 | not_friends |
| 333 | 222 | not_friends |
| 222 | 111 | friends |
| 111 | 222 | friends |
| 333 | 111 | not_friends |

event_rsvp table:

| user_id | event_id | event_type | attendance_status | event_date |
|---|---|---|---|---|
| 111 | 567 | public | going | 2022-07-12 |
| 222 | 789 | private | going | 2022-07-15 |
| 333 | 789 | private | maybe | 2022-07-15 |
| 111 | 234 | private | not_going | 2022-07-18 |
| 222 | 234 | private | going | 2022-07-18 |
| 333 | 234 | private | going | 2022-07-18 |

---

### Example Output

| user_a_id | user_b_id |
|---|---|
| 222 | 333 |
| 333 | 222 |

**Explanation:**

**Shared Private Events (where they showed interest):**
- Event 789: Users 222 (going) and 333 (maybe) both interested ✓
- Event 234: Users 222 (going), 333 (going) both interested ✓
- **Total shared: 2 events**

**Friendship Status:**
- (222, 333): not_friends ✓
- (333, 222): not_friends ✓

**Result:** Both user pairs meet criteria: not friends + 2+ shared private events

---

### MySQL Solution

```sql
WITH private_events AS (
  SELECT user_id, event_id
  FROM event_rsvp
  WHERE attendance_status IN ('going', 'maybe')
    AND event_type = 'private'
)
SELECT 
  friends.user_a_id, 
  friends.user_b_id
FROM private_events AS events_1
INNER JOIN private_events AS events_2
  ON events_1.user_id != events_2.user_id
  AND events_1.event_id = events_2.event_id
INNER JOIN friendship_status AS friends
  ON events_1.user_id = friends.user_a_id
  AND events_2.user_id = friends.user_b_id
WHERE friends.status = 'not_friends'
GROUP BY friends.user_a_id, friends.user_b_id
HAVING COUNT(*) >= 2
ORDER BY friends.user_a_id, friends.user_b_id;
```

---

### Query Explanation

**CTE (private_events):**
- Filter to private events where attendance_status IN ('going', 'maybe')

**Self-Join (events_1 ⨝ events_2):**
- Conditions: `events_1.event_id = events_2.event_id` AND `events_1.user_id != events_2.user_id`
- Creates all user pairs attending the same private events

**Join with friendship_status:**
- Match pairs: `events_1.user_id = user_a_id` AND `events_2.user_id = user_b_id`
- Filter: `WHERE friends.status = 'not_friends'`

**Group & Count:**
- `GROUP BY user_a_id, user_b_id`
- `HAVING COUNT(*) >= 2` keeps pairs with 2+ shared private events

---

### Key Concepts

- **Self-Join:** `INNER JOIN table AS t1 ON table AS t2` finds pairs within the same dataset
- **Filtering in Joins:** Apply conditions (different users, same event) directly in the ON clause
- **Multi-Table Join:** Combines overlaps (from events) with relationship status
- **HAVING >= 2:** Filters grouped pairs based on aggregate (shared event count)
- **Unordered Pairs:** The self-join naturally produces both (A,B) and (B,A) if both exist in friendship_status

---

## Question 7: 3-Topping Pizzas

**Source:** McKinsey SQL Interview Question

---

### Problem Description

A pizza chain is running a promotion where all 3-topping pizzas are sold at a fixed price. Given a list of available toppings with their individual costs, generate all possible 3-topping pizza combinations and calculate their total cost.

This is a **combinations problem**: find all unique sets of 3 toppings (where order doesn't matter).

---

### Example Input

pizza_toppings table:

| topping_name | ingredient_cost |
|---|---|
| Pepperoni | 0.50 |
| Sausage | 0.70 |
| Chicken | 0.55 |
| Extra Cheese | 0.40 |

### Example Output

| pizza | total_cost |
|---|---|
| Chicken,Pepperoni,Sausage | 1.75 |
| Chicken,Extra Cheese,Sausage | 1.65 |
| Extra Cheese,Pepperoni,Sausage | 1.60 |
| Chicken,Extra Cheese,Pepperoni | 1.45 |

**Explanation:**

All possible 3-topping combinations:
1. Chicken + Pepperoni + Sausage = 0.55 + 0.50 + 0.70 = **1.75** ✓
2. Chicken + Extra Cheese + Sausage = 0.55 + 0.40 + 0.70 = **1.65** ✓
3. Extra Cheese + Pepperoni + Sausage = 0.40 + 0.50 + 0.70 = **1.60** ✓
4. Chicken + Extra Cheese + Pepperoni = 0.55 + 0.40 + 0.50 = **1.45** ✓

Sorted: Highest cost first, then alphabetically by pizza name.

---

### MySQL Solution

```sql
SELECT 
  CONCAT(p1.topping_name, ',', p2.topping_name, ',', p3.topping_name) AS pizza,
  p1.ingredient_cost + p2.ingredient_cost + p3.ingredient_cost AS total_cost
FROM pizza_toppings AS p1
INNER JOIN pizza_toppings AS p2
  ON p1.topping_name < p2.topping_name 
INNER JOIN pizza_toppings AS p3
  ON p2.topping_name < p3.topping_name 
ORDER BY total_cost DESC, pizza ASC;
```

---

### Query Explanation

**Self-Joins with Ordering:**
- First join: `p1 < p2` generates pairs in alphabetical order
- Second join: `p2 < p3` extends to triplets  
- Conditions enforce: p1 < p2 < p3 alphabetically

**Result:** All unique 3-tuples in alphabetical order (no duplicates like A,B,C and B,A,C)

**Calculate & Sort:**
- `CONCAT()` creates pizza names from three toppings
- Sum individual costs for each combination
- `ORDER BY total_cost DESC, pizza ASC` sorts by cost (descending), then name (alphabetically)

---

### Key Concepts

- **Combinations vs Permutations:** `<` operators generate combinations (unordered), not permutations (ordered)
- **Self-Join with Ordering:** Joining a table to itself with inequality conditions is a classic pattern for generating combinations
- **Alphabetical Enforcement:** Comparison operators on strings ensure both uniqueness and alphabetical ordering in output
- **Avoiding Duplicates:** Without `<` conditions, permutations like (A,B,C) and (B,A,C) would both appear
- **CONCAT:** Builds readable output by joining multiple fields with a delimiter

---

## Question 8: Follow-Up AirPods Percentage

**Source:** Apple SQL Interview Question

---

### Problem Description

Apple's retention team wants to understand buyer behavior: what percentage of customers who bought iPhones also bought AirPods as their very next purchase (with no other purchases in between)?

This requires: (1) identifying all iPhone buyers, (2) checking if their next purchase is AirPods, (3) calculating the percentage.

---

### Example Input

transactions table:

| transaction_id | customer_id | product_name | transaction_timestamp |
|---|---|---|---|
| 1 | 101 | iPhone | 2022-08-08 00:00:00 |
| 2 | 101 | AirPods | 2022-08-08 00:00:00 |
| 5 | 301 | iPhone | 2022-09-05 00:00:00 |
| 6 | 301 | iPad | 2022-09-06 00:00:00 |
| 7 | 301 | AirPods | 2022-09-07 00:00:00 |

---

### Example Output

| follow_up_percentage |
|---|
| 50 |

**Explanation:**

**iPhone buyers:** 2 (customers 101, 301)

**Follow-up AirPods purchases:**
- Customer 101: iPhone → AirPods (same timestamp, consecutive) ✓
- Customer 301: iPhone → iPad → AirPods (NOT consecutive, iPad in between) ✗

**Result:** 1 out of 2 iPhone buyers bought AirPods next = 1/2 × 100 = **50%**

---

### MySQL Solution

```sql
-- Step 1: Get all iPhone buyers
WITH iphone_buyers AS (
  SELECT DISTINCT customer_id
  FROM transactions
  WHERE LOWER(product_name) = 'iphone'
),

-- Step 2: Check if their next purchase is AirPods
lag_products AS (
  SELECT
    customer_id,
    product_name,
    LAG(product_name) OVER (
      PARTITION BY customer_id
      ORDER BY transaction_timestamp, transaction_id
    ) AS prev_product
  FROM transactions
),

-- Step 3: Find iPhone buyers who bought AirPods next
airpod_after_iphone AS (
  SELECT DISTINCT customer_id
  FROM lag_products
  WHERE LOWER(product_name) = 'airpods'
    AND LOWER(prev_product) = 'iphone'
)

-- Step 4: Calculate percentage
SELECT
  ROUND(
    COUNT(DISTINCT aai.customer_id) * 100.0
    / COUNT(DISTINCT ib.customer_id),
    0
  ) AS follow_up_percentage
FROM iphone_buyers ib
LEFT JOIN airpod_after_iphone aai
  ON ib.customer_id = aai.customer_id;
```

---

### Query Explanation

**CTE 1 (iphone_buyers):** Get all distinct iPhone buyers (denominator base)

**CTE 2 (lag_products):** 
- `LAG()` with `PARTITION BY customer_id ORDER BY timestamp, transaction_id` 
- Gets previous product for each transaction in customer's history
- "Directly after" = current product is AirPods AND previous product is iPhone

**CTE 3 (airpod_after_iphone):** Filter to customers where AirPods immediately follows iPhone

**Main Query:**
- `LEFT JOIN iphone_buyers ⨝ airpod_after_iphone` keeps all iPhone buyers
- Numerator: `COUNT(DISTINCT airpod_after_iphone.customer_id)` (non-NULL)
- Denominator: `COUNT(DISTINCT iphone_buyers.customer_id)` (all)
- Result: Percentage = numerator / denominator × 100

---

### Critical Fixes from Original Query

⚠️ **The original provided query had several bugs:**

1. **GROUP BY before LAG():** Window functions must operate on raw rows, not aggregated groups
2. **Wrong Denominator:** Counted all customers instead of just iPhone buyers
3. **No Secondary Sort:** Same-timestamp purchases could produce non-deterministic ordering
4. **Readability:** Complex nesting made logic hard to verify

✅ **New query fixes:**
- No GROUP BY in the window function CTE
- Explicit iphone_buyers CTE for clear denominator
- Secondary sort by transaction_id for consistency
- Separate, labeled CTEs for each logical step

---

### Key Concepts

- **LAG():** Window function to access the previous row's value within a partition
- **PARTITION BY:** Groups customer transactions separately
- **ORDER BY:** Ensures transactions are in chronological order; secondary sort breaks ties
- **"Directly After":** Requires the previous product to be iPhone, not any product before it
- **Denominator Clarity:** Percentage base must be "iPhone buyers," not "all customers"
- **LEFT JOIN Logic:** Ensures all iPhone buyers are counted; NULL values indicate non-followers

