# Apache Airflow — ML Orchestration Skeleton
## Interview Reference: ZenML-to-Airflow Translation

> **Goal**: Demonstrate conceptual fluency in Airflow architecture and ML pipeline design.
> Sushant's production stack is ZenML — use this document as the translation layer.

---

## 1. What Airflow Is (and Isn't)

**What Airflow is**: Apache Airflow is an open-source **workflow orchestration platform** that lets you author, schedule, and monitor multi-step pipelines as DAGs (Directed Acyclic Graphs) of tasks — in pure Python. It does not move data or run compute; it tells other systems to do work and records what happened.

**Airflow vs ZenML — analogous but not equivalent**:

| | Airflow | ZenML |
|---|---|---|
| **Scope** | General-purpose (ETL, ML, data, any workflow) | ML-first (training, evaluation, deployment) |
| **Scheduling** | Built-in cron scheduler | External trigger (CI/CD, API, manual) |
| **Artifact handling** | Manual — XCom metadata + external storage paths | Native — auto-versioned typed artifacts per step |
| **ML tooling** | Requires manual MLflow / registry integration | Native integrations (MLflow, KServe, ONNX) |
| **Abstraction** | DAG of operators / tasks | Pipeline of typed steps |

**Are they analogous?** At the structural level — yes. Both represent workflows as a directed graph of discrete units with dependency edges. At the purpose level — no. Airflow is a general orchestrator that happens to run ML pipelines; ZenML is an ML platform that happens to orchestrate. ZenML can even use Airflow as its backend execution engine.

**What it solves**: Multi-step ML workflows (load → clean → train → evaluate → promote) that need scheduling, dependency management, retry logic, and observability — things cron jobs and shell scripts cannot provide cleanly at scale.

---

## 2. Architecture — The Four Components

```
┌──────────────┐     parses DAGs      ┌──────────────────┐
│  Scheduler   │ ───────────────────▶ │  Metadata DB     │
│              │ ◀─── task states ─── │  (PostgreSQL)    │
└──────┬───────┘                      └──────────────────┘
       │ dispatches                          ▲
       ▼                                     │ reads
┌──────────────┐    executes tasks    ┌──────┴───────┐
│   Executor   │ ───────────────────▶ │  Worker(s)   │
└──────────────┘                      └──────────────┘
                                             ▲
┌──────────────┐    reads UI from DB          │
│  Web Server  │                      (runs actual code)
└──────────────┘
```

| Component | Role |
|---|---|
| **Scheduler** | Parses DAG files, marks tasks ready, sends to Executor |
| **Executor** | Dispatches tasks to workers (Local / Celery / Kubernetes) |
| **Worker** | Runs the actual Python/Bash code |
| **Metadata DB** | Stores all run history, task states, XCom values — source of truth |
| **Web Server** | UI + REST API; reads from metadata DB |

**Key insight**: The Scheduler never runs your code. Workers do. The Scheduler only decides *when* and *in what order*.

### Kubernetes Primer — What an Interviewer Expects You to Know

Kubernetes (K8s) is a **container orchestration platform** — it manages the deployment, scaling, and lifecycle of containerised applications across a cluster of machines. Think of it as an operating system for a fleet of servers.

**Core concepts:**

| Concept | What it is |
|---|---|
| **Container** | A lightweight, isolated process bundling code + dependencies (via Docker). Runs the same everywhere. |
| **Pod** | The smallest deployable unit in Kubernetes. Wraps one or more containers that share networking and storage. One pod = one task in KubernetesExecutor. |
| **Node** | A physical or virtual machine in the cluster that runs pods. Nodes have CPU/RAM that pods consume. |
| **Cluster** | The full set of nodes managed together by Kubernetes. One control plane (master) + many worker nodes. |
| **Namespace** | Logical isolation within a cluster — e.g., `airflow`, `ml-training`, `monitoring` namespaces share hardware but are isolated in access and quotas. |
| **Deployment** | A spec declaring desired state — "run 3 replicas of this container image." Kubernetes ensures this is always true. |
| **Service** | A stable network endpoint (IP + DNS name) for a set of pods. Pods come and go; the Service address doesn't change. |
| **ConfigMap / Secret** | Inject configuration or credentials into pods at runtime without hardcoding in the image. |
| **Resource Requests/Limits** | Each pod declares how much CPU/RAM it needs (`requests`) and the maximum it can use (`limits`). Kubernetes uses this to schedule pods onto nodes that have capacity. |

**Where Helm Charts come in:**

Deploying Airflow (or any complex app) onto Kubernetes means writing dozens of YAML manifests — Deployments, Services, ConfigMaps, Secrets, PersistentVolumes. This is tedious and error-prone to manage manually.

**Helm** is the **package manager for Kubernetes**. A **Helm Chart** is a pre-packaged, parameterisable bundle of all the Kubernetes manifests needed to deploy an application.

```
Without Helm:  write 15+ YAML files manually → kubectl apply each one → repeat per environment
With Helm:     helm install airflow apache-airflow/airflow --set executor=KubernetesExecutor
```

| Concept | What it is |
|---|---|
| **Chart** | A directory of templated Kubernetes manifests for one application (e.g., the official `apache-airflow` chart) |
| **values.yaml** | The configuration file where you override defaults — number of workers, executor type, image tag, resource limits |
| **Release** | One deployed instance of a chart. You can have `airflow-dev` and `airflow-prod` as two releases of the same chart |
| **`helm install`** | Deploys the chart to the cluster, creating all K8s resources at once |
| **`helm upgrade`** | Updates a release — e.g., bumps Airflow version or changes executor config without rewriting manifests |
| **`helm rollback`** | Reverts a release to a previous version — one command undoes a bad deploy |

**Where Helm fits in the deployment chain:**

```
Docker image (your code)
      ↓ pushed to
ECR / DockerHub (image registry)
      ↓ referenced in
Helm Chart values.yaml
      ↓ deployed via
helm install / upgrade
      ↓ creates
Kubernetes resources (Pods, Services, ConfigMaps...)
      ↓ managed by
Kubernetes cluster
```

In practice: the official Apache Airflow Helm chart packages the Scheduler, Webserver, Workers, and PostgreSQL metadata DB as one deployable unit. Your team only needs to override `values.yaml` — executor type, image tag, resource requests — and `helm upgrade` handles the rest. No manual pod management.

**How Kubernetes relates to Airflow:**

With `KubernetesExecutor`, every Airflow task runs in its own dedicated pod — created when the task starts, deleted when it finishes. This means:
- A heavy training task gets a 16GB RAM pod; a lightweight logging task gets 512MB — no contention.
- Failed pods don't affect other tasks — full isolation.
- The cluster auto-scales: if 20 tasks are queued, Kubernetes spins up 20 pods in parallel (subject to node capacity).

**The interview framing:**
> "Kubernetes is the infrastructure layer beneath the orchestration layer. Airflow decides *what* to run and *when*; Kubernetes decides *where* to run it and ensures it gets the right resources. In a production ML platform, Airflow's KubernetesExecutor bridges the two — each task becomes a pod spec, and Kubernetes handles placement, resource allocation, and cleanup."

---

## 3. DAG — The Core Concept

A DAG is a Python file that defines a **directed, acyclic graph of tasks**. No loops — tasks flow forward only.

```python
from airflow.decorators import dag
from datetime import datetime, timedelta

@dag(
    dag_id="kpi_model_retraining",
    schedule="0 3 * * *",        # cron: 3am daily
    start_date=datetime(2024, 1, 1),
    catchup=False,                # ← critical in production
    max_active_runs=1,            # ← prevent concurrent training runs
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": True,
    },
)
def kpi_retraining():
    ...
```

### Three parameters that matter most

**`catchup=False`**
If you deploy a DAG with `start_date` 30 days ago and `catchup=True` (default), Airflow queues 30 simultaneous backfill runs — flooding your workers. Always set `catchup=False` unless backfilling is intentional.

**`max_active_runs=1`**
Prevents a second scheduled run from starting while the previous one is still running. Essential for training pipelines — you never want two jobs simultaneously writing to the same model registry entry.

**`schedule` and logical date**
Airflow runs at the *end* of an interval, not the start. A daily DAG with `schedule="@daily"` and `start_date=2024-01-01` triggers its first run on `2024-01-02` for the logical date `2024-01-01`. Use `{{ ds }}` in templates to get `data_interval_start` as `YYYY-MM-DD`.

---

## 4. Tasks — The Unit of Work

A task = one step in the pipeline. Three ways to define one:

### TaskFlow API (preferred — Airflow 2.x)

```python
from airflow.decorators import task

@task
def load_data(ds=None) -> str:           # ds injected from Airflow context
    path = fetch_from_vertica(ds)
    save_to_s3(path)
    return path                          # return value auto-pushed to XCom

@task
def train_model(data_path: str) -> dict: # argument = auto-pull from XCom
    df = pd.read_parquet(data_path)
    metrics = fit_and_save(df)
    return metrics
```

### Classic Operator (for non-Python work)

```python
from airflow.operators.bash import BashOperator

spark_job = BashOperator(
    task_id="run_spark_features",
    bash_command="spark-submit /jobs/features.py --date {{ ds }}",
)
```

### Sensor (waits for external condition)

```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

wait_for_data = S3KeySensor(
    task_id="wait_for_snapshot",
    bucket_name="data-lake",
    bucket_key="snapshots/{{ ds_nodash }}/",
    poke_interval=300,
    timeout=7200,
    mode="reschedule",    # release worker slot between polls — always use for long waits
)
```

### Wiring dependencies

```python
# Bitshift operators
extract >> transform >> train >> evaluate

# Fan-out (parallel)
train >> [evaluate_train, evaluate_valid]

# Fan-in (merge)
[load_table_a, load_table_b] >> join_step
```

---

## 5. XCom — Passing Data Between Tasks

XCom (Cross-Communication) stores values in the **metadata DB**. It is for metadata, not data.

```
Rule: XCom the PATH to data, never the data itself.
```

```python
@task
def train(data_path: str) -> dict:
    model_path = "s3://models/model_2024_01_01.pkl"
    fit_and_save(data_path, model_path)
    return {"model_path": model_path, "rmse": 0.12}   # ✓ small dict

# NOT this:
@task
def train_bad(df: pd.DataFrame) -> pd.DataFrame:      # ✗ DataFrames don't belong in XCom
    return df.transform(...)
```

**Why**: XCom is stored in PostgreSQL. A 100MB DataFrame serialised there kills your metadata DB. The pattern is: write large data to S3/GCS, XCom the path, downstream task reads from the path.

---

## 6. Complete ML Pipeline Skeleton

```python
from airflow.decorators import dag, task
from airflow.operators.python import ShortCircuitOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from datetime import datetime, timedelta

@dag(
    dag_id="kpi_model_daily_retraining",
    schedule="0 3 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
)
def kpi_retraining():

    # 1. Wait for upstream data (sensor pattern)
    wait = S3KeySensor(
        task_id="wait_for_snapshot",
        bucket_name="data-lake",
        bucket_key="snapshots/{{ ds_nodash }}/",
        mode="reschedule",
        poke_interval=300,
        timeout=7200,
    )

    # 2. Quality gate — abort early if data is bad
    @task
    def validate(ds=None) -> str:
        count = get_row_count(ds)
        if count < 10_000:
            raise ValueError(f"Insufficient data: {count} rows")
        path = f"s3://staging/clean_{ds}.parquet"
        save_cleaned_data(ds, path)
        return path                            # XCom: path, not DataFrame

    # 3. Tune + Train (separate tasks so failures are isolated)
    @task
    def tune(data_path: str) -> dict:
        df = pd.read_parquet(data_path)
        return optuna_search(df)               # XCom: best params dict

    @task
    def train(data_path: str, best_params: dict, ds=None) -> dict:
        df = pd.read_parquet(data_path)
        model_path = f"s3://models/kpi_{ds}.pkl"
        fit_and_save(df, best_params, model_path)
        return {"model_path": model_path}

    # 4. Evaluate and promote
    @task
    def evaluate(model_info: dict, ds=None) -> dict:
        metrics = run_evaluation(model_info["model_path"], ds)
        mlflow.log_metrics(metrics)
        return metrics

    @task
    def promote(metrics: dict, model_info: dict):
        champion = get_champion_metrics()
        if metrics["rmse"] < champion["rmse"]:
            register_champion(model_info["model_path"])

    # 5. Always-run failure alert
    alert = PythonOperator(
        task_id="alert_on_failure",
        python_callable=send_pagerduty,
        trigger_rule="one_failed",             # runs even if upstream fails
    )

    # Wire
    path    = validate()
    params  = tune(path)
    model   = train(path, params)
    metrics = evaluate(model)
    promote(metrics, model)

    wait >> path
    [model, metrics] >> alert

dag_instance = kpi_retraining()
```

**What this demonstrates**:
- Sensor for upstream data dependency
- Quality gate with early abort
- Isolated steps (tune failure doesn't lose cleaned data)
- XCom by path
- Always-on failure alerting via `trigger_rule`
- Implicit dependencies from TaskFlow call order

---

## 7. Key Patterns to Know

### Branching

```python
def choose_path(ds=None, **ctx) -> str:
    return "train_full" if get_row_count(ds) > 100_000 else "train_lite"

branch = BranchPythonOperator(task_id="branch", python_callable=choose_path)
merge  = EmptyOperator(task_id="merge", trigger_rule="none_failed_min_one_success")

branch >> [train_full, train_lite] >> merge
# Unchosen branch is SKIPPED, not FAILED — merge needs the trigger_rule override
```

### Trigger Rules

| Rule | When it runs |
|---|---|
| `all_success` (default) | All upstream succeeded |
| `one_failed` | At least one upstream failed — use for alerts |
| `none_failed_min_one_success` | None failed, one succeeded — use after branching |
| `all_done` | All upstream finished regardless of state |

### Parallelism via `expand` (Airflow 2.3+)

```python
@task
def train_kpi(config: dict) -> dict:
    return run_training(config)

configs = [{"goal": "ctr"}, {"goal": "ecpc"}, {"goal": "vcpm"}]
results = train_kpi.expand(config=configs)   # 3 parallel task instances at runtime
```

### Executors — One-line summary each

| Executor | Use when |
|---|---|
| `LocalExecutor` | Single machine, moderate load |
| `CeleryExecutor` | Multi-worker, large-scale, many small tasks |
| `KubernetesExecutor` | Cloud-native; each task gets a dedicated pod with its own resource spec |

**KubernetesExecutor is the right answer for ML**: heavy training tasks get 16GB RAM pods; lightweight monitoring tasks get 512MB pods — no resource contention.

---

## 8. Airflow vs ZenML — Your Translation Layer

| Concept | ZenML (your stack) | Airflow equivalent |
|---|---|---|
| Workflow | `@pipeline` decorated function | DAG (Python file) |
| Unit of work | `@step` function | `@task` / Operator |
| Step ordering | Implicit from function calls | `>>` operator or TaskFlow call order |
| Data passing | Typed Artifacts (versioned, S3-backed) | XCom (metadata only) + external storage |
| Scheduling | CI/CD trigger (GitHub Actions) | Built-in Scheduler with cron |
| Config | YAML → `pipeline.with_options(config_path=...)` | Variables, `op_kwargs`, Connections |
| Experiment tracking | Native MLflow integration | Manual via MLflow hook in task |
| Model promotion | `promote_model` step → MLflow registry | Task calling MLflow / custom registry API |
| Conditional execution | Not native | `BranchPythonOperator`, `ShortCircuitOperator` |
| Parallelism | Parallel branches in pipeline graph | Fan-out tasks, `expand()` |
| Backfill | Manual pipeline re-run | `airflow dags backfill -s DATE -e DATE` |

### The key difference to articulate

ZenML auto-versions every step's input and output as a named artifact — you can reproduce any past run by loading the exact artifact versions. Airflow doesn't version artifacts natively; you manage this by embedding dates in S3 paths (`model_2024_01_01.pkl`). MLflow or DVC fills that gap when using Airflow for ML.

---

## 9. Interview Talking Points

**"What is Airflow and how does it work?"**
> "Airflow is a metadata-driven workflow orchestrator. The Scheduler parses DAG files and marks tasks ready when their dependencies are met. The Executor dispatches those tasks to Workers, which run the actual code. Everything is recorded in a PostgreSQL metadata database. Critically, Airflow doesn't move data — it tells other systems to do work. That separation is what makes it composable: the same Airflow DAG can orchestrate Spark jobs, Python scripts, and dbt runs within one dependency graph."

**"Walk me through how you'd design a daily retraining pipeline."**
> "I'd structure it as a DAG with `catchup=False` and `max_active_runs=1` so concurrent runs can't race on the model registry. First, an S3 sensor in `reschedule` mode waits for the upstream data snapshot — this decouples the training DAG from the data pipeline DAG. Then a validation task that aborts early if row count is below threshold. Then separate tune and train tasks — keeping them separate means a training failure doesn't force hyperparameter search to re-run. Artifacts flow between tasks as S3 paths via XCom, never as DataFrames. A final alert task with `trigger_rule='one_failed'` fires PagerDuty if anything breaks."

**"How do you pass data between tasks?"**
> "Via XCom, but only for small metadata. XCom is backed by the metadata database — you don't want a DataFrame going in there. The production pattern is: write data to S3, XCom the path, downstream task reads from S3. This also means you can restart a failed task from the middle of the pipeline without re-running the data loading step — the path is already in XCom."

**"How does this relate to your ZenML experience?"**
> "The mental model is identical — a directed graph of typed steps with dependency edges. The differences are in scheduling and artifact management. ZenML doesn't have a built-in scheduler; we trigger via GitHub Actions CI. Airflow has a mature scheduler with cron support and backfill capability, which is a real advantage for time-series workflows. ZenML auto-versions artifacts; in Airflow you manage versioning yourself through path conventions or MLflow. If I were porting our KPI model training system to Airflow, it would be a DAG with task groups per KPI, `expand()` for parallelism across eight models, and the same MLflow logging calls we already use — the Python logic is identical, just the orchestration wrapper changes."

**"When would you use KubernetesExecutor?"**
> "Any ML platform where tasks have very different resource profiles. A data cleaning task needs 2 cores and 2GB RAM; a hyperparameter search needs 8 cores and 32GB. With CeleryExecutor, workers are sized for the heaviest task — most workers are wasteful most of the time. KubernetesExecutor creates a pod per task with its own resource spec, so resources are allocated exactly to what each task needs. The tradeoff is pod startup latency — 30-60 seconds — so for DAGs with many fast tasks, Celery is faster."
