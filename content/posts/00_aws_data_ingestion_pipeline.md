---
title: "AWS Data Ingestion Pipeline — Production Architecture"
date: 2026-07-19T00:00:00+05:30
draft: true
tags: ["aws", "data-engineering", "iot", "interview-prep"]
summary: "High-level walkthrough of an AWS-based IoT data ingestion pipeline for a manufacturing use case, and how the core services connect end-to-end."
---

# AWS Data Ingestion Pipeline — Production Architecture
## Interview Reference: High-Level AWS Services & How They Connect

> **Source**: Tvarit (previous role) — IoT manufacturing data ingestion pipeline.
> **Goal**: Demonstrate production deployment fluency. No deep API dives — architecture + reasoning.

---

## The Big Picture

Two types of data sources → one ingestion backbone → two storage zones → downstream ML.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  DATA SOURCES                                                                ║
║                                                                              ║
║  PLC / Edge Devices         SQL / SCADA / MES Databases                     ║
║  (publish via MQTT)         (queried via Python script on-prem)             ║
╚══════════╤═══════════════════════════════╤════════════════════════════════════╝
           │ MQTT                          │ MQTT (via script → IoT Core)
           ▼                               ▼
╔══════════════════════════════════════════════════════════════╗
║  AWS IoT Core  (MQTT Broker)                                 ║
║  - Authenticates devices via Certificates + Policies         ║
║  - Routes messages via IoT Rules → trigger Lambda            ║
╚══════════════════════════════════╤═══════════════════════════╝
                                   │ triggers
                    ┌──────────────┴─────────────┐
                    │                             │
                    ▼                             ▼
         ┌──────────────────┐         ┌──────────────────────┐
         │  Lambda          │         │  Lambda              │
         │  (Landing Zone)  │         │  (Timestamp Monitor) │
         │  Writes raw msg  │         │  Writes last-seen    │
         │  to S3           │         │  JSON to S3, alerts  │
         └────────┬─────────┘         │  if data goes stale  │
                  │                   └──────────────────────┘
                  ▼
╔═════════════════════════════╗
║  S3 — Landing Zone          ║  ← raw, per-topic, per-message files
║  (one prefix per topic)     ║
╚══════════════╤══════════════╝
               │ daily trigger via Lambda
               ▼
╔══════════════════════════════════════════════════════════════╗
║  AWS Batch  (Docker container → Papermill → PySpark)         ║
║  - Reads landing zone                                        ║
║  - Consolidates / aggregates across topics                   ║
║  - Evolved from EMR (see design decisions)                   ║
╚══════════════╤═══════════════════════════════════════════════╝
               │
               ▼
╔═════════════════════════════╗
║  S3 — Consolidation Zone    ║  ← cleaned, aggregated, analysis-ready
╚═════════════════════════════╝

Infrastructure provisioned via CloudFormation (IoT Policy + Thing + Rule per customer)
Container images stored in ECR, pulled by Batch at runtime
```

---

## Service Glossary — What Each Does Here

| Service | One-line role | In this pipeline |
|---|---|---|
| **AWS IoT Core** | Managed MQTT broker for device-to-cloud messaging | Receives raw sensor data from edge devices; routes to Lambda via Rules |
| **MQTT** | Message Queuing Telemetry Transport — lightweight pub/sub protocol where devices publish messages to named topic paths and all subscribers receive them; designed for low-bandwidth, unreliable networks | Devices publish to topic paths like `Plant/Line/Machine/Sensor`; IoT Core brokers it |
| **IoT Thing** | Cloud record representing one physical device | Each customer device is registered here with a certificate for auth |
| **IoT Policy** | Access control for what a Thing can publish/subscribe to | `topic/#` means the device can publish to any topic path |
| **IoT Rule** | Conditional logic on incoming messages → trigger action | SQL-like filter on topic → fires Lambda when message arrives |
| **AWS Lambda** | Serverless function, event-driven, max 15 min | Three uses: (1) write to S3 landing zone, (2) monitor data freshness, (3) trigger Batch jobs |
| **S3** | Object storage | Two zones: landing (raw files per message) and consolidation (processed aggregates) |
| **AWS Batch** | Run containerised jobs of arbitrary duration | Runs PySpark consolidation jobs that exceed Lambda's 15-min limit |
| **ECR** | Docker image registry (like DockerHub, inside AWS) | Stores the container image that Batch pulls to run consolidation notebooks |
| **AWS EMR** | Managed Spark cluster | Earlier approach for consolidation — one cluster per plant per day; replaced by Batch |
| **CloudFormation** | Infrastructure-as-Code; define AWS resources in YAML | Automates customer onboarding — one template creates IoT Policy + Thing + Rule + Topic |
| **Parameter Store** | Key-value store for configs and secrets in AWS | Stores connection strings, timestamps, and configs referenced by Lambda at runtime |
| **IAM Role** | Permissions for AWS services to call other services | Lambda needs an IAM role to write to S3; Batch needs a role to pull from ECR |
| **CloudWatch** | Logs and metrics for AWS services | Lambda execution logs; Batch job logs; cron triggers for monitoring Lambda |

---

## The Two Ingestion Paths

### Path 1 — IoT / Edge Devices (PLC data)

```
Edge device (PLC / Advantech)
  → publishes MQTT message to topic: Gutersloh/Foundry/Datec/3EM06_v2
    → AWS IoT Core receives it
      → IoT Rule matches topic
        → triggers Lambda (landing zone writer)
          → Lambda writes raw JSON to S3: s3://landing/Gutersloh/Foundry/Datec/3EM06_v2/{random_int}.json
```

**Authentication**: Each device holds a certificate + private key downloaded at provisioning. IoT Core validates the cert — no username/password. Certificates stored in `s3://iot-certificates.tvarit.com`.

**Topic naming**: `Plant/Line/Department/SensorID` — hierarchical, mirrors the physical plant layout.

**Random int suffix on S3 key**: Prevents race conditions when multiple messages arrive in the same second. Without it, concurrent Lambda invocations overwrite each other.

---

### Path 2 — SQL / SCADA / MES Data

```
On-prem SQL/SCADA DB
  → Python script (running on Tvarit server → customer RDP)
    → queries DB for time window [last_timestamp, now]
      → reads rows, serialises as CSV
        → publishes each row as MQTT message to IoT Core
          → same Lambda path as Path 1 → lands in S3
```

**Why push through IoT Core?** Unifies both ingestion paths into one downstream pipeline. The consolidation zone doesn't need to know whether data came from a PLC or a SQL database.

**Timestamp tracking**: A `timestamp.json` file per topic stores the last successfully queried window. The script reads it to determine the query start time — avoids re-ingesting already-processed data.

---

## The Consolidation Flow

```
S3 Landing Zone (raw, per-message files)
  ↓
Lambda (cron: daily) → submits job to AWS Batch
  ↓
AWS Batch pulls Docker image from ECR
  → Container runs bash script → runs Papermill notebook → PySpark job
    → reads all files from landing zone prefix
    → aggregates across topics / machines / time
    → writes to S3 Consolidation Zone
```

**Papermill**: A Python library that executes Jupyter notebooks programmatically with injected parameters. The consolidation logic lives in a notebook; Papermill passes plant/date configs at runtime without hardcoding.

---

## Key Design Decisions

### Why Lambda → Batch (not Lambda → Lambda)?

| | Lambda | AWS Batch |
|---|---|---|
| Max runtime | **15 minutes** | Hours / no hard limit |
| Trigger | Event-driven | Job queue |
| Scaling | Auto-scales | Configurable compute environment |
| Cost | Per 100ms | Per vCPU/hr while running |

**The problem**: Consolidating a full day of IoT data across all topics and plants can take 30–90 minutes. Lambda cannot do this. The fix: Lambda's only job is to *submit* the Batch job and return immediately. Batch handles the long-running computation.

**Pattern**: Lambda = lightweight dispatcher. Batch = heavy worker.

### Why Batch → not EMR?

EMR creates a full Spark cluster (master + worker nodes) for each run — spin-up time is 5–10 minutes even before the job starts. For consolidation jobs that run once daily per plant, this overhead is significant and costly. AWS Batch with a Docker container running PySpark is lighter: one container, no cluster orchestration overhead, faster startup. EMR is better suited for truly large-scale, multi-TB Spark jobs where cluster parallelism is the bottleneck.

### Why CloudFormation for customer onboarding?

Before CloudFormation, adding a new customer required manually clicking through the IoT console to create Policy → Thing → Rule → Topic. Error-prone and not auditable. CloudFormation defines the same set of resources as a YAML template — one command creates everything, one command tears it all down. The same template is parameterised per customer (plant name, certificate ARN, topic prefix).

---

## Monitoring — Data Freshness Alerting

```
CloudWatch cron (every 1hr)
  → triggers Lambda (timestamp monitor)
    → for each topic: reads timestamp.json from S3
      → computes: now - last_modified_timestamp
        → if > 1hr: publish to SNS → email/Slack alert
```

**Why this matters**: In IoT pipelines, **silent failures** are the most dangerous failure mode. A broken MQTT connection or offline PLC doesn't throw an error — data just stops arriving. The timestamp monitor catches this by checking that each topic received data within the expected window.

---

## How Services Are Connected (Permission Layer)

```
IoT Core → Lambda        (IoT Rule Action — IAM permissions)
Lambda   → S3            (Lambda execution role — s3:PutObject)
Lambda   → Batch         (Lambda execution role — batch:SubmitJob)
Batch    → ECR           (Batch compute role — ecr:GetAuthorizationToken)
Batch    → S3            (Batch job role — s3:GetObject, s3:PutObject)
Lambda   → CloudWatch    (auto-granted — logs:CreateLogGroup, logs:PutLogEvents)
```

Every service-to-service call in AWS is governed by an IAM Role. The common interview trap: "why can't Lambda write to S3?" — because the Lambda execution role doesn't have `s3:PutObject` on the target bucket. IAM is the connective tissue of the whole architecture.

---

## Interview Talking Points

**"Walk me through a data pipeline you've built in production."**
> "At Tvarit I built an IoT data ingestion pipeline on AWS. Edge devices at manufacturing plants published sensor readings via MQTT to AWS IoT Core. IoT Rules routed each incoming message to a Lambda function that wrote raw JSON to an S3 landing zone — one file per message, per topic, with a random suffix to prevent concurrent writes from overwriting each other. For SQL/SCADA data that couldn't push natively, we ran a Python script on-prem that queried the database and published each row through the same IoT Core pipeline, giving us a single unified ingestion path. Daily consolidation ran as an AWS Batch job — a Docker container pulling a Papermill-parameterised PySpark notebook from ECR, aggregating the landing zone into a clean consolidation zone in S3. Customer onboarding was fully automated via CloudFormation — one YAML template provisioned the IoT Policy, Thing, Rule, and topic structure, making new plant setup a single CLI command."

**"How did you handle the Lambda 15-minute limit?"**
> "The consolidation step needed 30–90 minutes to aggregate a full day of sensor data across all plant topics. Lambda can't do that. The pattern we used was: Lambda as a lightweight dispatcher — its only job is to call `batch.submit_job()` and return immediately. AWS Batch picks up the job from the queue and runs it in a Docker container for as long as needed. Lambda stays well within its limit; Batch handles the heavy lifting."

**"How did you monitor data quality in the IoT pipeline?"**
> "Silent failures are the worst failure mode in IoT — a broken MQTT connection just means data stops arriving, no error anywhere. We handled this with a freshness monitor: a Lambda function on an hourly CloudWatch cron that read a `timestamp.json` file we maintained per topic in S3. Whenever a message was processed, we updated that file with the current timestamp. If the difference between now and the last timestamp exceeded one hour, we fired an alert. This gave us proactive detection of offline devices or broken connections before anyone noticed missing data downstream."

**"What's CloudFormation and why did you use it?"**
> "CloudFormation is AWS's Infrastructure-as-Code service — you define your resources in a YAML template and AWS provisions and manages them. We used it for customer onboarding: adding a new manufacturing plant used to require manually creating an IoT Policy, Thing, Rule, and topic structure through the console. With CloudFormation, the same set of resources is defined once as a parameterised template. Onboarding a new plant is one CLI command; deleting a plant is one stack delete. It's auditable, repeatable, and eliminates the console-click errors that used to cause us incidents."
