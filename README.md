# AI Healthcare Interaction Embodiment Assistant

A production-grade, multi-agent AI platform for **Multiple Sclerosis (MS) risk screening** built on the principles of Agentic AI Interaction and Embodiment. The system orchestrates five specialized AI agents through a clinically-governed pipeline that screens patient populations, computes explainable risk scores, and takes autonomous or semi-autonomous clinical actions — all under configurable safety guardrails.

The platform implements two open interoperability standards — **Model Context Protocol (MCP)** and **Agent-to-Agent (A2A)** — enabling external AI systems to interact with the screening pipeline through standardized interfaces.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Multi-Agent Pipeline](#multi-agent-pipeline)
- [Technology Stack](#technology-stack)
- [Services & Ports](#services--ports)
- [Data Model](#data-model)
- [API Reference](#api-reference)
- [MCP & A2A Protocols](#mcp--a2a-protocols)
- [Frontend Application](#frontend-application)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Governance & Safety](#governance--safety)
- [Analytics & Fairness](#analytics--fairness)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Key Features

- **Multi-Agent Screening Pipeline** — Five sequential agents (Retrieval, Phenotyping, Notes & Imaging, Safety, Coordinator) process patient populations through a clinically-validated screening workflow
- **Explainable Risk Scoring** — Per-feature contribution weights, decision rationale, and patient risk cards for every assessment
- **Tiered Autonomy** — Four action levels (No Action, Recommend Neuro Review, Draft MRI Order, Auto-Order MRI) governed by configurable policy thresholds
- **Safety Governance** — Real-time safety flags (PHI detection, low evidence, minor patient, contradictions), rate limiting on auto-actions, and automatic autonomy downgrade when flags are present
- **Fairness Analysis** — Built-in subgroup analysis by sex, age band, and diagnosis with demographic disparity detection
- **What-If Policy Simulation** — Interactive policy threshold editor showing real-time impact on action distribution, precision, and recall
- **MCP Integration** — 8 tools and 5 resources exposed via Model Context Protocol for LLM and agent interoperability
- **A2A Gateway** — Agent-to-Agent protocol enabling external AI agents to discover, communicate with, and orchestrate screening agents
- **Real-Time Updates** — WebSocket channels for live workflow progress and notification streaming
- **Comprehensive Audit Trail** — Every system action logged with actor, target, timestamp, and detailed metadata
- **Compliance Reporting** — Auto-generated fairness, safety, performance, and full compliance reports per workflow run

---

## Architecture Overview

The system is composed of **9 Docker services** organized into four architectural tiers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT TIER                                  │
│  React SPA (TypeScript + Material-UI)              Port 3055       │
│  13 Pages | Zustand State | Recharts | WebSocket Client            │
├─────────────────────────────────────────────────────────────────────┤
│                      GATEWAY TIER                                   │
│  Nginx Reverse Proxy ──> Node.js BFF Gateway       Port 4055      │
│  Routes: /api -> Backend | /mcp -> MCP | /a2a -> A2A               │
├─────────────────────────────────────────────────────────────────────┤
│                     SERVICES TIER                                   │
│  Django + Daphne ASGI (REST + WebSocket)           Port 8055       │
│  Celery Worker (queues: default, agents, analytics)                │
│  Celery Beat (DatabaseScheduler)                                   │
│  MCP Server (8 tools, 5 resources)                 Port 9055       │
│  A2A Gateway (5 agents, task orchestration)         Port 9155       │
├─────────────────────────────────────────────────────────────────────┤
│                       DATA TIER                                     │
│  PostgreSQL 16 (ms_risk_lab)                       Port 5455       │
│  Redis 7 (cache slot 0, broker slot 1, gateway slot 2) Port 6399  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Interaction**: User clicks "Run Screening Workflow" on the Dashboard
2. **API Request**: React -> Nginx -> Node.js Gateway -> Django REST API
3. **Task Queuing**: Django creates a Celery task and returns `task_id`
4. **Agent Execution**: Celery Worker executes the 5-agent pipeline sequentially
5. **Persistence**: Risk assessments, workflow metrics, and notifications saved to PostgreSQL
6. **Real-Time Update**: WebSocket pushes workflow completion to frontend
7. **Audit**: Every action logged to the audit trail

---

## Multi-Agent Pipeline

The screening workflow executes five specialized agents in sequence:

### 1. Retrieval Agent

Identifies candidate patients from the full population using four mandatory gates:

| Gate | Criteria | Purpose |
|------|----------|---------|
| MRI Gate | `mri_lesions == True` | Patient has brain MRI with lesions |
| Notes Gate | `note_has_ms_terms == True` | Clinical notes contain MS-related terminology |
| Symptom Gate | `symptom_count >= 2` | At least 2 of 8 core neurological symptoms |
| Visit Gate | `visits_last_year >= 6` | Sufficient clinical engagement for screening |

**Output**: List of candidate patient IDs and per-gate counts.

### 2. Phenotyping Agent V2

Computes a risk score (0.0 - 1.0) using weighted feature contributions:

| Feature | Weight | Category |
|---------|--------|----------|
| Optic neuritis | 0.22 | Core symptom |
| MRI lesions | 0.18 | Imaging |
| Paresthesia | 0.14 | Core symptom |
| Weakness | 0.13 | Core symptom |
| Note MS terms | 0.12 | NLP |
| Gait instability | 0.10 | Core symptom |
| Vertigo | 0.08 | Core symptom |
| Fatigue | 0.06 | Core symptom |
| Bladder issues | 0.05 | Core symptom |
| Cognitive fog | 0.04 | Core symptom |

**Extended V2 features**: Vitamin D deficiency bonus (+0.03), infectious mono history (+0.02), smartform neuro symptom score (scaled), PATHS-like function score (scaled).

**Lookalike diagnosis penalties**: Migraine (-0.08), B12 deficiency (-0.10), anxiety (-0.05), fibromyalgia (-0.07), stroke/TIA (-0.12).

**Output**: `risk_score`, `feature_contributions` dictionary.

### 3. Notes & Imaging Agent

Analyzes clinical notes for MS-specific and non-MS terminology:

- **MS phrases**: demyelinating, periventricular lesions, oligoclonal bands, optic neuritis, relapsing symptoms, neurology referral, MRI brain w/wo contrast
- **Non-MS phrases**: tension headache, vitamin deficiency, stress-related, poor sleep, peripheral neuropathy, viral illness, benign positional vertigo

**Output**: `note_ms_terms_flag`, `note_nonms_terms_flag`, `ms_terms_found[]`, `nonms_terms_found[]`, `note_excerpt`.

### 4. Safety & Governance Agent

Applies safety checks and generates flags:

| Flag | Trigger | Severity |
|------|---------|----------|
| `PHI_DETECTED` | Note contains pattern `name:` | Critical |
| `LOW_EVIDENCE_CASE` | Fewer than 2 active symptoms | Warning |
| `MINOR_PATIENT` | Age < 18 | Warning |
| `HIGH_RISK_LOW_EVIDENCE` | Risk >= 0.80 AND < 2 symptoms | Critical |

**Output**: `flags[]`, `flag_count`.

### 5. Coordinator

Makes the final clinical action decision based on policy thresholds:

| Risk Score Range | Action | Autonomy Level |
|-----------------|--------|----------------|
| `< 0.65` | `NO_ACTION` | `RECOMMEND_ONLY` (0) |
| `0.65 - 0.80` | `RECOMMEND_NEURO_REVIEW` | `RECOMMEND_ONLY` (1) |
| `0.80 - 0.90` | `DRAFT_MRI_ORDER` | `DRAFT_ORDER` (2) |
| `>= 0.90` | `AUTO_ORDER_MRI_AND_NOTIFY_NEURO` | `AUTO_ORDER_WITH_GUARDRAILS` (3) |

**Safety Override**: If `flag_count > 0` and autonomy level >= 2 (DRAFT or AUTO), the action is downgraded to `RECOMMEND_NEURO_REVIEW`.

**Rate Limiting**: Auto-order actions are capped at `max_auto_actions_per_day` (default: 20). Excess auto-orders are downgraded to `DRAFT_MRI_ORDER`.

**Output**: `action`, `autonomy_level`, `rationale[]`.

---

## Technology Stack

### Backend

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Primary backend language |
| Django | 5.x | Web framework and ORM |
| Django REST Framework | 3.15+ | REST API |
| Daphne | Latest | ASGI server (HTTP + WebSocket) |
| Django Channels | 4.x | WebSocket support with Redis channel layer |
| Celery | 5.x | Distributed task queue |
| django-celery-beat | Latest | Periodic task scheduling |
| pandas | 2.x | Data manipulation for agent pipeline |
| WhiteNoise | Latest | Static file serving |
| django-cors-headers | Latest | CORS support |

### Frontend

| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18.x | UI framework |
| TypeScript | 5.x | Type-safe JavaScript |
| Material-UI (MUI) | 6.3+ | Component library |
| React Router | 6.x | SPA routing |
| Zustand | Latest | State management |
| Recharts | Latest | Charts and data visualization |
| Axios | Latest | HTTP client |
| Notistack | Latest | Toast notifications |
| date-fns | Latest | Date formatting |
| Vite | 5.x | Build tool |

### Infrastructure

| Technology | Version | Purpose |
|-----------|---------|---------|
| Docker | Latest | Containerization |
| Docker Compose | v2 | Multi-service orchestration |
| PostgreSQL | 16 Alpine | Primary database |
| Redis | 7 Alpine | Cache, message broker, channel layer |
| Nginx | Alpine | Reverse proxy and SPA serving |
| Node.js | 20 Alpine | API gateway (BFF pattern) |

---

## Services & Ports

| Service | Container Name | Internal Port | External Port | Description |
|---------|---------------|---------------|---------------|-------------|
| PostgreSQL | ms_risk_postgres | 5432 | 5455 | Primary database |
| Redis | ms_risk_redis | 6379 | 6399 | Cache and message broker |
| Django Backend | ms_risk_backend | 8000 | 8055 | REST API + WebSocket |
| Celery Worker | ms_risk_celery_worker | -- | -- | Async task execution |
| Celery Beat | ms_risk_celery_beat | -- | -- | Scheduled tasks |
| MCP Server | ms_risk_mcp_server | 9000 | 9055 | Model Context Protocol |
| A2A Gateway | ms_risk_a2a_gateway | 9100 | 9155 | Agent-to-Agent Protocol |
| Node.js Gateway | ms_risk_node_gateway | 4000 | 4055 | Backend-for-Frontend |
| React Frontend | ms_risk_frontend | 80 | 3055 | Web application |

**Network**: All services communicate over the `ms_risk_network` bridge network.

**Volumes**:
- `postgres_data` — PostgreSQL data persistence
- `redis_data` — Redis data persistence
- `backend_static` — Django collected static files

---

## Data Model

### Core Models

#### Patient

Synthetic patient records with 30+ clinical fields:

```
Patient
  id (UUID, primary key)
  patient_id (CharField, unique, e.g. "P00001")
  Demographics: age, sex
  Clinical: visits_last_year, lookalike_dx
  Symptoms (8 boolean flags):
    optic_neuritis, paresthesia, weakness, gait_instability
    vertigo, fatigue, bladder_issues, cognitive_fog
  Imaging: has_mri, mri_lesions
  NLP: note (text), note_has_ms_terms
  Ground Truth: true_at_risk (boolean)
  Lab Markers: vitamin_d_ngml, vitamin_d_deficient, infectious_mono_history
  Computed Scores: smartform_neuro_symptom_score, paths_like_function_score
  Timestamps: created_at, updated_at
```

#### RiskAssessment

Per-patient, per-workflow risk evaluation:

```
RiskAssessment
  id (UUID), patient (FK), run_id (UUID -> WorkflowRun)
  Scoring: risk_score (0-1), action, autonomy_level
  Interpretability: feature_contributions (JSON), rationale (JSON)
  Safety: flags (JSON), flag_count
  NLP: notes_analysis (JSON), llm_summary
  Snapshot: patient_card (JSON)
  Review: reviewed_by, review_notes, reviewed_at
```

#### WorkflowRun

Tracks a single execution of the screening pipeline:

```
WorkflowRun
  id (UUID), policy (FK -> PolicyConfiguration)
  Status: PENDING -> RUNNING -> COMPLETED / FAILED
  Counts: total_patients, candidates_found, flagged_count
  Metrics: precision, recall, safety_flag_rate
  Actions: auto_actions, draft_actions, recommend_actions
  Performance: duration_seconds, error_message
```

#### PolicyConfiguration

Configurable thresholds and rate limits:

```
PolicyConfiguration
  name, is_active, created_by
  Thresholds: risk_review (0.65), draft_order (0.80), auto_order (0.90)
  Rate Limits: max_auto_actions_per_day (20)
```

### Supporting Models

| Model | Purpose |
|-------|---------|
| `AuditLog` | Action trail: action_type, actor, target_type, target_id, details (JSON) |
| `Notification` | Alerts: title, message, severity, category, related_patient_id, metadata |
| `GovernanceRule` | Safety rules: rule_type, condition (JSON), severity, is_active |
| `ComplianceReport` | Generated reports: report_type, workflow_run (FK), data (JSON) |
| `AgentExecution` | Agent logs: agent_name, patient_id_ref, payload (JSON), duration_ms |
| `SystemConfiguration` | Runtime config: key-value pairs |

---

## API Reference

### REST API Endpoints

All endpoints are prefixed with `/api/`.

#### Patients

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/patients/` | List patients (filterable by sex, lookalike_dx, true_at_risk, has_mri) |
| GET | `/api/patients/{id}/` | Patient detail with risk history |
| GET | `/api/patients/{id}/risk_history/` | Historical risk assessments |
| GET | `/api/patients/summary/` | Population statistics |

#### Risk Assessments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/assessments/` | List assessments (filterable by action, autonomy_level, run_id) |
| GET | `/api/assessments/{id}/` | Assessment detail |
| POST | `/api/assessments/{id}/review/` | Submit clinician review |
| GET | `/api/assessments/high_risk/` | High-risk assessments (query: threshold, run_id) |
| GET | `/api/assessments/pending_review/` | Assessments awaiting review |

#### Policies

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/policies/` | List all policies |
| POST | `/api/policies/` | Create new policy |
| PUT | `/api/policies/{id}/` | Update policy |
| POST | `/api/policies/{id}/activate/` | Set as active policy |

#### Workflows

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/workflows/` | List workflow runs |
| GET | `/api/workflows/{id}/` | Run detail |
| POST | `/api/workflows/trigger/` | Start new screening run |
| GET | `/api/workflows/{id}/metrics/` | Comprehensive run metrics |
| GET | `/api/workflows/{id}/risk_distribution/` | Risk score histogram |
| GET | `/api/workflows/{id}/action_distribution/` | Action type breakdown |
| GET | `/api/workflows/{id}/autonomy_distribution/` | Autonomy level breakdown |
| GET | `/api/workflows/{id}/calibration/` | Risk score calibration |
| GET | `/api/workflows/{id}/fairness/` | Fairness analysis (query: group_by) |

#### Governance & Compliance

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/governance-rules/` | List governance rules |
| POST | `/api/governance-rules/` | Create rule |
| PUT | `/api/governance-rules/{id}/` | Update rule |
| DELETE | `/api/governance-rules/{id}/` | Delete rule |
| GET | `/api/compliance-reports/` | List compliance reports |
| POST | `/api/compliance-reports/generate/` | Generate new report (async) |

#### Audit & Notifications

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/audit-logs/` | List audit logs (filterable by action_type, actor) |
| GET | `/api/notifications/` | List notifications |
| POST | `/api/notifications/{id}/mark_read/` | Mark notification as read |
| POST | `/api/notifications/mark_all_read/` | Mark all as read |
| GET | `/api/notifications/unread_count/` | Unread notification count |

#### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard/` | Aggregated dashboard data |
| POST | `/api/what-if/` | Policy what-if simulation |

### WebSocket Endpoints

| Endpoint | Purpose |
|----------|---------|
| `ws://host:8055/ws/workflow/` | Real-time workflow progress updates |
| `ws://host:8055/ws/notifications/` | Live notification streaming |

---

## MCP & A2A Protocols

### Model Context Protocol (MCP)

The MCP server exposes the screening platform's capabilities to external LLMs and AI agents.

**Base URL**: `http://host:9055/mcp/`

#### Tools (8)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `screen_patient` | Run single-patient screening | `patient_id` |
| `run_screening_workflow` | Execute full population screening | `policy_id`, `patient_limit` |
| `get_patient_risk_card` | Generate explainable risk card | `patient_id` |
| `analyze_fairness` | Demographic fairness analysis | `run_id`, `group_by` |
| `what_if_policy` | Simulate threshold changes | `risk_review`, `draft_order`, `auto_order` |
| `review_assessment` | Submit clinician review | `assessment_id`, `reviewer`, `notes` |
| `get_workflow_metrics` | Comprehensive run metrics | `run_id` |
| `summarize_note` | Clinical note summarization | `patient_id` |

#### Resources (5)

| URI | Description |
|-----|-------------|
| `msrisk://patients` | Patient registry access |
| `msrisk://assessments` | Risk assessment data |
| `msrisk://policies` | Policy configurations |
| `msrisk://governance` | Governance rules |
| `msrisk://audit_logs` | Audit trail records |

### Agent-to-Agent Protocol (A2A)

The A2A gateway enables direct agent-to-agent communication and orchestration.

**Base URL**: `http://host:9155/a2a/`

#### Registered Agents

| Agent ID | Name | Capabilities |
|----------|------|-------------|
| `retrieval-agent` | Retrieval Agent | Patient retrieval, candidate selection, EHR search |
| `phenotyping-agent` | Phenotyping Agent | Risk scoring, feature analysis |
| `notes-imaging-agent` | Notes & Imaging Agent | Note analysis, imaging interpretation |
| `safety-governance-agent` | Safety & Governance Agent | Safety checks, compliance verification |
| `coordinator-agent` | Coordinator Agent | Decision making, autonomy determination |

#### Task Lifecycle

```
POST /a2a/tasks/create/       -> status: "pending"
POST /a2a/tasks/{id}/execute/ -> status: "running" -> "completed" / "failed"
GET  /a2a/tasks/{id}/         -> Get task result
POST /a2a/orchestrate/        -> Full pipeline orchestration
GET  /a2a/agents/             -> List all registered agents
GET  /a2a/agents/{agent_id}/  -> Get agent card (capabilities, schema)
```

---

## Frontend Application

### Pages (13)

| Page | Route | Description |
|------|-------|-------------|
| **Dashboard** | `/` | Overview: patient stats, latest run metrics, workflow history, action/autonomy charts |
| **Patients** | `/patients` | Searchable patient list with sex, diagnosis, MRI status filters |
| **Patient Detail** | `/patients/:id` | Demographics, symptoms, lab markers, risk history, A2A screening trigger |
| **Assessments** | `/assessments` | Risk assessment results with action/autonomy filtering, flag summaries |
| **Workflows** | `/workflows` | Workflow run history with status indicators |
| **Workflow Detail** | `/workflows/:id` | Run metrics, risk distribution, action/autonomy charts, agent execution timeline |
| **Fairness** | `/fairness` | Subgroup analysis by sex, age band, diagnosis |
| **What-If** | `/what-if` | Interactive policy threshold sliders with real-time impact preview |
| **Policies** | `/policies` | Create, edit, and activate policy configurations |
| **Governance** | `/governance` | Manage governance rules, view and open compliance reports |
| **Agents** | `/agents` | Agent execution logs and performance metrics |
| **Audit Trail** | `/audit` | Comprehensive audit log with filters and clickable detail dialogs |
| **Notifications** | `/notifications` | Alert center with severity icons, patient ID chips, mark-as-read |

### Key UI Components

| Component | Description |
|-----------|-------------|
| `Layout` | Navigation sidebar with 13 menu items, header with dark mode toggle |
| `StatCard` | KPI display card with icon, value, subtitle, and accent color |
| `RiskScoreBadge` | Color-coded risk score (green < 0.5, orange 0.5-0.8, red > 0.8) |
| `ActionBadge` | Action type chip (NO_ACTION, RECOMMEND, DRAFT, AUTO) |
| `AutonomyBadge` | Autonomy level chip with color coding |

### State Management

**Zustand store** manages:
- Dashboard data and loading state
- Unread notification count
- Dark mode preference

### Charts and Visualizations

Built with **Recharts**:
- Pie charts for action distribution
- Bar charts for autonomy levels, feature contributions, subgroup metrics
- Histograms for risk score distribution
- Calibration plots (predicted vs. actual)

---

## Getting Started

### Prerequisites

- Docker and Docker Compose v2
- At least 4 GB free RAM
- Ports 3055, 4055, 5455, 6399, 8055, 9055, 9155 available

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-healthcare-embodiment
   ```

2. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env to set OPENAI_API_KEY for LLM features (optional)
   ```

3. **Start all services**:
   ```bash
   docker compose up --build -d
   ```

4. **Wait for initialization** (first run seeds 2,500 synthetic patients):
   ```bash
   # Watch backend startup (migrations, seed data)
   docker compose logs -f backend

   # Wait until you see "Listening on TCP address 0.0.0.0:8000"
   ```

5. **Access the application**:
   - **Web UI**: http://localhost:3055 (or http://108.48.39.238:3055)
   - **REST API**: http://localhost:8055/api/
   - **MCP Server**: http://localhost:9055/mcp/
   - **A2A Gateway**: http://localhost:9155/a2a/

6. **Run your first screening workflow**:
   - Open the Dashboard at http://localhost:3055
   - Click **"Run Screening Workflow"**
   - Watch results populate on the Workflows, Assessments, and Notifications pages

### Stopping Services

```bash
docker compose down          # Stop all services (preserves data)
docker compose down -v       # Stop and remove volumes (full reset)
```

### Rebuilding After Code Changes

```bash
docker compose up --build -d
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | `MSRisk2024Secure!` | PostgreSQL password |
| `DJANGO_SECRET_KEY` | (generated) | Django secret key for cryptographic signing |
| `DEBUG` | `0` | Django debug mode (set to 1 for development) |
| `OPENAI_API_KEY` | (empty) | OpenAI API key for LLM-powered note summarization |
| `OPENAI_BASE_URL` | (empty) | Custom OpenAI-compatible endpoint URL |
| `SITE_URL` | `http://108.48.39.238:3055` | Public-facing site URL |

### Policy Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_review_threshold` | 0.65 | Minimum risk score to recommend neuro review |
| `draft_order_threshold` | 0.80 | Minimum risk score to draft MRI order |
| `auto_order_threshold` | 0.90 | Minimum risk score for autonomous MRI order |
| `max_auto_actions_per_day` | 20 | Daily rate limit on autonomous actions |

### Celery Configuration

| Setting | Value |
|---------|-------|
| Broker | Redis (database slot 1) |
| Result Backend | Django ORM (database-backed) |
| Task Queues | default, agents, analytics |
| Worker Concurrency | 4 processes |
| Task Time Limit | 300 seconds |
| Scheduler | django_celery_beat DatabaseScheduler |
| Serialization | JSON |

### Seed Data

On first startup, the backend seeds **2,500 synthetic patients** with:
- Realistic age/sex distributions
- 8 neurological symptom flags per patient
- Lookalike diagnoses (migraine, B12 deficiency, anxiety, fibromyalgia, stroke/TIA)
- Clinical notes with embedded MS and non-MS terminology
- MRI status and lesion flags
- Extended lab markers (Vitamin D, mono history)
- Ground truth `true_at_risk` labels for metrics validation

---

## Governance & Safety

### Safety Flags

The Safety & Governance Agent applies four automated checks on every candidate patient:

| Flag | Condition | Severity | Effect |
|------|-----------|----------|--------|
| `PHI_DETECTED` | Clinical note matches regex `name:` | Critical | Flag for manual review |
| `LOW_EVIDENCE_CASE` | Fewer than 2 active symptoms | Warning | Flag for manual review |
| `MINOR_PATIENT` | Patient age < 18 | Warning | Flag for manual review |
| `HIGH_RISK_LOW_EVIDENCE` | Risk >= 0.80 AND < 2 symptoms | Critical | Flag + autonomy downgrade |

### Safety Override Mechanism

When **any** safety flag is present and the Coordinator's autonomy level is `DRAFT_ORDER` or `AUTO_ORDER_WITH_GUARDRAILS`, the action is automatically downgraded to `RECOMMEND_NEURO_REVIEW`. This ensures all flagged patients require explicit human clinician review before any clinical action is taken.

### Rate Limiting

Auto-order actions (`AUTO_ORDER_MRI_AND_NOTIFY_NEURO`) are capped at the active policy's `max_auto_actions_per_day` (default: 20). When the daily limit is reached within a workflow run, subsequent auto-eligible patients receive `DRAFT_MRI_ORDER` instead, requiring manual approval.

### Governance Rules

Configurable rules in the Governance page:

| Rule Type | Description |
|-----------|-------------|
| `PHI_CHECK` | Detect protected health information in notes |
| `EVIDENCE_CHECK` | Minimum evidence requirements |
| `DEMOGRAPHIC_CHECK` | Age and demographic safeguards |
| `CONTRADICTION_CHECK` | High risk + low evidence contradiction |
| `RATE_LIMIT` | Action frequency limits |
| `CUSTOM` | User-defined governance rules |

### Audit Trail

Every significant system action is recorded with full context:

| Action Type | Logged Details |
|-------------|---------------|
| `AGENT_RUN` | Policy ID, policy name, patient limit, task ID |
| `ASSESSMENT_REVIEW` | Reviewer, notes, assessment ID |
| `POLICY_CHANGE` | Old/new values, changed_by |
| `RULE_CHANGE` | Rule modifications, activated/deactivated |
| `DATA_ACCESS` | MCP/A2A resource access with requester info |

### Compliance Reports

Four report types generated per workflow run:

| Type | Contents |
|------|----------|
| `FAIRNESS` | Subgroup analysis by sex, age band, diagnosis with disparity metrics |
| `SAFETY` | Safety flag rates, override counts, PHI detection summary |
| `PERFORMANCE` | Precision, recall, F1, confusion matrix, calibration |
| `FULL` | Comprehensive report combining fairness, safety, and performance |

---

## Analytics & Fairness

### Workflow Metrics

After each workflow run completes, the system automatically computes:

- **Precision & Recall**: Using `true_at_risk` ground truth labels
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True Positives, False Positives, True Negatives, False Negatives
- **Risk Distribution**: Histogram with mean, median, standard deviation, and quartiles
- **Action Distribution**: Count per action type (No Action, Recommend, Draft, Auto)
- **Autonomy Distribution**: Count per autonomy level
- **Calibration**: Predicted risk vs. actual outcome by decile bins
- **Safety Flag Rate**: Percentage of assessments with one or more safety flags

### Fairness Analysis

Subgroup metrics are computed across three demographic dimensions:

| Dimension | Groups |
|-----------|--------|
| Sex | Male, Female |
| Age Band | <30, 30-39, 40-49, 50-59, 60+ |
| Diagnosis | Migraine, B12 deficiency, Anxiety, Fibromyalgia, Stroke/TIA, None |

Per-subgroup metrics include:
- **Flagged rate** — Proportion screened as at-risk
- **Average risk score** — Mean predicted risk
- **Auto/draft action rate** — Higher-autonomy action proportion
- **Safety flag rate** — Proportion with safety concerns
- **True at-risk rate** — Ground truth prevalence
- **MRI rate** — Proportion with MRI imaging

### What-If Policy Simulation

The What-If page provides interactive policy exploration:
- Adjust `risk_review_threshold`, `draft_order_threshold`, and `auto_order_threshold` with sliders
- See real-time recalculation of action distribution, precision, recall, and confusion matrix
- Compare proposed thresholds against the current active policy
- No data is modified — simulation runs on existing assessment scores

---

## Project Structure

```
ai-healthcare-embodiment/
├── docker-compose.yml              # 9-service orchestration
├── .env.example                    # Environment template
├── README.md                       # This file
├── docs/
│   ├── architecture-diagram.drawio # Technical architecture (draw.io)
│   └── technical-architecture.pptx # Architecture presentation (PowerPoint)
│
├── backend/
│   ├── Dockerfile                  # Python backend image
│   ├── requirements.txt            # Python dependencies
│   ├── manage.py                   # Django management
│   ├── config/
│   │   ├── settings.py             # Django settings (DB, cache, celery, channels)
│   │   ├── urls.py                 # Root URL configuration
│   │   ├── asgi.py                 # ASGI application (Daphne + Channels)
│   │   └── celery.py              # Celery app configuration
│   │
│   ├── core/                       # Core app
│   │   ├── models.py              # AuditLog, Notification, SystemConfiguration
│   │   ├── consumers.py           # WebSocket consumers
│   │   └── management/commands/
│   │       └── seed_data.py       # Generate 2500 synthetic patients
│   │
│   ├── patients/                   # Patients app
│   │   └── models.py              # Patient, RiskAssessment, PolicyConfiguration, WorkflowRun
│   │
│   ├── agents/                     # AI Agents app
│   │   ├── models.py              # AgentExecution
│   │   ├── base.py                # BaseAgent abstract class, AgentRegistry
│   │   ├── retrieval.py           # RetrievalAgent (4-gate candidate selection)
│   │   ├── phenotyping.py         # PhenotypingAgentV2 (weighted risk scoring)
│   │   ├── notes_imaging.py       # NotesImagingAgent (NLP term extraction)
│   │   ├── safety.py              # SafetyGovernanceAgent (flag generation)
│   │   ├── coordinator.py         # Coordinator (action + autonomy decision)
│   │   ├── workflow.py            # Full pipeline orchestration + notifications
│   │   └── tasks.py               # Celery task wrapper
│   │
│   ├── governance/                 # Governance app
│   │   └── models.py              # GovernanceRule, ComplianceReport
│   │
│   ├── analytics/                  # Analytics app
│   │   ├── services.py            # Metrics: precision, recall, fairness, calibration
│   │   └── tasks.py               # Compliance report generation task
│   │
│   ├── mcp/                        # Model Context Protocol
│   │   ├── protocol.py            # MCPServer, MCPTool, MCPResource, MCPMessage
│   │   ├── views.py               # MCP HTTP endpoints (tools, resources, invoke)
│   │   └── urls.py                # MCP URL routing
│   │
│   ├── a2a/                        # Agent-to-Agent Protocol
│   │   ├── protocol.py            # A2AGateway, AgentCard, A2ATask
│   │   ├── views.py               # A2A HTTP endpoints (agents, tasks, orchestrate)
│   │   └── urls.py                # A2A URL routing
│   │
│   └── api/                        # REST API app
│       ├── views.py               # ViewSets: Patient, Assessment, Policy, Workflow, etc.
│       ├── serializers.py         # DRF serializers with nested relations
│       └── urls.py                # API URL routing (DefaultRouter)
│
└── frontend/
    ├── Dockerfile                  # Multi-stage build (Vite build -> Nginx serve)
    ├── nginx.conf                  # Nginx config (SPA routing, API/MCP/A2A proxy)
    ├── package.json                # Node.js dependencies
    ├── vite.config.ts              # Vite build configuration
    ├── tsconfig.json               # TypeScript configuration
    │
    ├── gateway/
    │   ├── Dockerfile              # Node.js gateway image
    │   ├── server.js               # Express + http-proxy-middleware (BFF)
    │   └── package.json            # Gateway dependencies
    │
    └── src/
        ├── App.tsx                 # React Router configuration (13 routes)
        ├── main.tsx                # Application entry point
        ├── theme.ts                # Material-UI custom theme
        ├── types/index.ts          # TypeScript interfaces (15+ types)
        ├── services/api.ts         # Axios API client (30+ endpoints)
        ├── store/index.ts          # Zustand state store
        ├── utils/helpers.ts        # Formatting utilities
        ├── components/
        │   ├── Layout.tsx          # Navigation sidebar + header
        │   ├── RiskBadge.tsx       # Risk/action/autonomy badges
        │   └── StatCard.tsx        # KPI display card
        └── pages/
            ├── DashboardPage.tsx   # Overview dashboard with charts
            ├── PatientsPage.tsx    # Patient list with filters
            ├── PatientDetailPage.tsx # Patient detail + A2A screening
            ├── AssessmentsPage.tsx  # Risk assessments table
            ├── WorkflowsPage.tsx   # Workflow run history
            ├── WorkflowDetailPage.tsx # Run detail with visualizations
            ├── FairnessPage.tsx    # Subgroup fairness analysis
            ├── WhatIfPage.tsx      # Policy simulation with sliders
            ├── PoliciesPage.tsx    # Policy CRUD management
            ├── GovernancePage.tsx   # Rules + compliance report dialogs
            ├── AgentsPage.tsx      # Agent execution logs
            ├── AuditPage.tsx       # Audit trail with detail dialogs
            └── NotificationsPage.tsx # Notification center with patient IDs
```

---

## Troubleshooting

### Backend won't start

```bash
# Check logs for migration or startup errors
docker compose logs backend

# Common: port already in use
lsof -i :8055

# Common: database not ready yet (backend waits for healthcheck)
docker compose logs postgres
```

### Celery worker not processing tasks

```bash
# Check worker logs for connection or import errors
docker compose logs celery_worker

# Verify Redis is running and reachable
docker compose exec redis redis-cli ping
# Expected: PONG

# Check if tasks are queued
docker compose exec redis redis-cli LLEN celery
```

### Workflow stuck in RUNNING status

```bash
# Check worker logs for agent execution errors
docker compose logs celery_worker --tail=100

# Verify the task was received
docker compose logs celery_worker | grep "run_screening_workflow"
```

### Frontend shows empty data / blank pages

1. Ensure backend has finished seeding data:
   ```bash
   docker compose logs backend | grep "seed_data"
   ```
2. Verify API is reachable:
   ```bash
   curl http://localhost:8055/api/patients/?limit=1
   ```
3. Check browser developer console (F12) for CORS or network errors
4. Verify the Node.js gateway is running:
   ```bash
   docker compose logs node_gateway
   ```

### Notifications page is empty

Notifications are generated when a workflow completes. To see notifications:
1. Go to the Dashboard
2. Click **"Run Screening Workflow"**
3. Wait for the workflow to complete (check Workflows page)
4. Navigate to Notifications

### Assessments page is empty

Risk assessments are created during workflow execution. Run a screening workflow first from the Dashboard.

### Database reset (start fresh)

```bash
docker compose down -v       # Remove containers and volumes
docker compose up --build -d # Rebuild and restart (re-seeds 2500 patients)
```

### Container health checks

```bash
# Check health of all services
docker compose ps

# Expected: all services show "healthy" or "running"
```

---

## Documentation

Additional documentation is available in the `docs/` directory:

| File | Description |
|------|-------------|
| `architecture-diagram.drawio` | Full technical architecture diagram (open with draw.io or diagrams.net) |
| `technical-architecture.pptx` | Architecture presentation slides (PowerPoint) |

---

## License

This project is developed for research and educational purposes in healthcare AI interaction and embodiment.
