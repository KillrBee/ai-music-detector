# Multi-LLM Comparison Platform — Implementation Plan

**Status:** Draft for review
**Owner:** greg.broadhead@gmail.com
**Working name:** LLM Arena (subdirectory `llm-arena/` in this repo; can be split out later)

---

## 1. Vision & Goals

A single-user (initially) web application for interacting with **N LLM endpoints
simultaneously**. One chat box broadcasts a message to every selected endpoint;
each model's response renders in its own panel. Selected responses can be sent
to a "judge" LLM that evaluates quality, accuracy, and completeness, compares
responses against each other, ranks them, and identifies the domains where each
model succeeded or failed. Models don't just talk: tool calls **execute
autonomously** with full access to the host VM, each model working inside its
own subdirectory under the conversation's directory, so comparisons cover real
agentic work — not just prose. **Everything** — user prompts, system prompts, tool
calls and results, thinking/reasoning blocks, streamed deltas, final responses,
judge verdicts, token usage, latency — is durably logged, keyed by conversation
ID and tagged with model name + version, in a schema designed for downstream
analytical processing.

### Functional requirements (from the request)

| # | Requirement |
|---|-------------|
| R1 | Configure an arbitrary number (N) of LLM API connections (endpoints + credentials + model + params) |
| R2 | Single chat box; on send, the same message goes to every *selected* endpoint |
| R3 | Each endpoint's response lives in its own window/panel in the UI, streaming in real time |
| R4 | Select any subset of responses and dispatch them to a judge LLM for evaluation |
| R5 | Judge evaluates quality, accuracy, completeness; compares responses to each other; ranks them; maps domain capabilities and failures per model |
| R6 | All artifacts logged and stored by conversation ID, tagged with model name and version |
| R7 | Full-fidelity record: user prompts, system prompts, tool calls, tool results, thinking blocks, final responses — queryable for analytics |
| R8 | Tool calls execute autonomously (no user in the loop); each LLM has full access to the host VM but works inside its own subdirectory under the conversation directory |

### Non-goals (v1)

- Multi-user auth / teams / RBAC (design leaves room for it; don't build it)
- Fine-tuning or RAG pipelines
- Mobile-native apps (responsive web only)
- Billing management for provider accounts (we *record* cost estimates only)

---

## 2. High-Level Architecture

Same stack as the existing project (FastAPI backend, React + TypeScript + Vite +
Tailwind frontend) so tooling, linting, and deployment knowledge transfer directly.

```
┌────────────────────────────────────────────────────────────────────┐
│  Frontend (React + TS + Vite + Tailwind)                           │
│  ┌──────────┐ ┌───────────────┐ ┌───────────┐ ┌────────────────┐  │
│  │ Endpoint │ │ Chat Composer │ │ Response  │ │ Evaluation /   │  │
│  │ Manager  │ │ (broadcast)   │ │ Grid      │ │ Judge Console  │  │
│  └──────────┘ └───────────────┘ └───────────┘ └────────────────┘  │
│         REST (CRUD, history)  +  SSE (streaming deltas)            │
├────────────────────────────────────────────────────────────────────┤
│  Backend (FastAPI, async) — runs on a dedicated/disposable VM      │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Provider       │  │ Broadcast    │  │ Evaluation Service   │    │
│  │ Registry +     │  │ Orchestrator │  │ (judge pipeline)     │    │
│  │ Adapters       │  │ (agentic     │  │                      │    │
│  └───────┬────────┘  │  fan-out)    │  └──────────┬───────────┘    │
│          │           └──────┬───────┘             │                │
│  ┌───────┴──────────────────┴─────────────────────┴────────────┐  │
│  │ Tool Execution Runtime — full VM access, per-run workspaces │  │
│  │ data/conversations/{conv_id}/{model-slug}/                  │  │
│  └──────────────────────────────────────────────────────────────┘ │
│          └── normalized event stream ─────────────┘                │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Event Recorder → SQLite (WAL) via SQLAlchemy + raw JSONL   │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Credential Vault (encrypted at rest, never sent to client) │   │
│  └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
            │                    │                    │
     Anthropic API          OpenAI API      OpenAI-compatible (Ollama,
   (Messages API)      (Responses/Chat)     vLLM, OpenRouter, Groq, …)
                                            + Google Gemini
```

### Key architectural decisions

1. **Adapter pattern over a heavyweight framework.** Each provider gets a thin
   async adapter that translates to/from a **normalized internal event
   schema** (see §4). We deliberately avoid LangChain-style abstraction: the
   whole point of this tool is faithful capture of what each API actually sent
   and received, so adapters store the *raw* request/response alongside the
   normalized form. Provider SDKs used where they help (`anthropic`,
   `openai`); the OpenAI adapter doubles as the adapter for every
   OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, Groq, Together,
   LM Studio, etc.) via a configurable `base_url` — this is what makes
   "n-number of connections" cheap.

2. **SSE (Server-Sent Events) for streaming, not WebSockets.** The data flow is
   strictly server→client during generation; SSE is simpler, proxies well, and
   auto-reconnects. One SSE stream per broadcast carries multiplexed events
   tagged with `run_id`, so the frontend demuxes into panels. (WebSocket
   upgrade is a contained refactor later if bidirectional needs emerge.)

3. **SQLite (WAL mode) + append-only JSONL as v1 storage.** Single-user tool,
   zero-ops. Schema is written in SQLAlchemy with Alembic migrations so the
   move to Postgres is a connection-string change when/if needed. The JSONL
   event log is the belt-and-suspenders raw record (one file per day,
   one JSON object per event) that survives any schema migration mistake and
   is directly ingestible by DuckDB/pandas for analytics.

4. **Write-ahead recording.** The Event Recorder persists the outbound request
   *before* dispatch and every inbound delta/terminal event as it arrives. A
   crash mid-generation still leaves an accurate partial record with a
   `status = 'interrupted'` marker — no "the log is whatever the UI happened
   to show" problem.

5. **Autonomous tool execution; the VM is the sandbox.** Models don't just
   *emit* tool calls — the backend executes them and loops until the model
   finishes (an agentic loop per run, like a coding agent). Tools run with
   the backend's own privileges, i.e. full access to the host, so the
   deployment assumption is a **dedicated, disposable virtual machine**: the
   isolation boundary is the VM, not the application. Filesystem hygiene
   between concurrently-running models comes from **per-run workspaces**
   (`data/conversations/{conversation_id}/{model-slug}/`) — each model's cwd
   and instructed write-root is its own subdirectory. See §4.4.

6. **The judge is just another endpoint.** Evaluations are dispatched through
   the same adapter/recorder machinery as chat broadcasts, so judge prompts,
   judge thinking blocks, and judge outputs are logged with the same fidelity
   and tagged to the same conversation ID (R6/R7 apply to the judge too).

---

## 3. Repository Layout

New self-contained directory so the audio-detector app is untouched:

```
llm-arena/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app factory, CORS, lifespan
│   │   ├── config.py                # pydantic-settings (DB path, vault key, defaults)
│   │   ├── db/
│   │   │   ├── engine.py            # async SQLAlchemy engine/session
│   │   │   ├── models.py            # ORM models (schema in §5)
│   │   │   └── migrations/          # Alembic
│   │   ├── providers/
│   │   │   ├── base.py              # ProviderAdapter protocol + normalized events
│   │   │   ├── anthropic_adapter.py
│   │   │   ├── openai_adapter.py    # also serves all OpenAI-compatible endpoints
│   │   │   ├── google_adapter.py
│   │   │   └── registry.py          # endpoint CRUD, health checks, model listing
│   │   ├── tools/
│   │   │   ├── registry.py          # standard toolset + per-provider tool-def translation
│   │   │   ├── executor.py          # runs tool calls, streams output, enforces budgets
│   │   │   ├── workspace.py         # per-run workspace creation, manifest, diff snapshots
│   │   │   └── builtins/            # bash.py, read_file.py, write_file.py, web_fetch.py
│   │   ├── services/
│   │   │   ├── broadcast.py         # agentic fan-out orchestrator (asyncio.TaskGroup)
│   │   │   ├── recorder.py          # event persistence (DB + JSONL)
│   │   │   ├── evaluation.py        # judge pipeline, rubric prompts, verdict parsing
│   │   │   ├── vault.py             # Fernet-encrypted credential store
│   │   │   └── costing.py           # token→cost estimation tables
│   │   ├── routers/
│   │   │   ├── endpoints.py         # /api/endpoints CRUD + test
│   │   │   ├── conversations.py     # /api/conversations, messages, history
│   │   │   ├── broadcasts.py        # /api/broadcasts + SSE stream
│   │   │   ├── evaluations.py       # /api/evaluations + SSE stream
│   │   │   └── analytics.py         # /api/analytics/* + exports
│   │   └── schemas/                 # Pydantic request/response DTOs
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/                     # typed client + SSE hooks
│   │   ├── state/                   # Zustand stores (see §7)
│   │   ├── components/
│   │   │   ├── endpoints/           # EndpointManager, EndpointForm, ModelPicker
│   │   │   ├── chat/                # Composer, SystemPromptEditor, EndpointSelector
│   │   │   ├── responses/           # ResponseGrid, ResponsePanel, ThinkingBlock,
│   │   │   │                        #   ToolCallBlock, UsageFooter
│   │   │   ├── evaluation/          # JudgePicker, RubricEditor, VerdictView,
│   │   │   │                        #   RankingTable, DomainMatrix
│   │   │   └── history/             # ConversationList, ConversationReplay
│   │   └── types/                   # shared TS types mirroring backend schemas
│   └── package.json
└── README.md
```

---

## 4. Provider Abstraction & Normalized Event Schema

### 4.1 Adapter protocol

```python
class ProviderAdapter(Protocol):
    async def stream_chat(
        self,
        request: NormalizedRequest,          # messages, system, params, tools
    ) -> AsyncIterator[NormalizedEvent]: ...

    async def list_models(self) -> list[ModelInfo]: ...
    async def health_check(self) -> HealthStatus: ...
```

`NormalizedRequest` carries: ordered message history (already provider-agnostic),
system prompt, model id, sampling params (temperature, max_tokens, top_p),
optional tool definitions, and reasoning/thinking configuration.
Each adapter is responsible for:

- translating history into the provider's wire format (e.g., Anthropic's
  separate `system` field vs. OpenAI's `role: system` message);
- enabling capture of **thinking/reasoning blocks** where supported
  (Anthropic extended thinking, OpenAI reasoning summaries, DeepSeek-style
  `reasoning_content` on OpenAI-compatible endpoints) and emitting them as
  first-class events rather than discarding them;
- surfacing **tool-use requests** emitted by the model as events; the
  orchestrator hands them to the Tool Execution Runtime (§4.4), and the
  adapter feeds the results back into the provider's continuation format —
  the loop runs without user involvement;
- attaching the **raw provider payload** to every event for archival.

### 4.2 Normalized event types

Every adapter yields a uniform stream; this is the atom of the entire system —
the streamer, the recorder, the UI, and the analytics layer all consume it.

| `event_type` | Payload highlights |
|---|---|
| `run_started` | resolved model id, provider-reported model *version/snapshot*, request snapshot ref |
| `text_delta` | incremental assistant text |
| `thinking_delta` | incremental reasoning/thinking text (+ signature/redaction flags) |
| `tool_call` | tool name, call id, full arguments JSON |
| `tool_output_delta` | incremental stdout/stderr while a tool executes (live terminal feel) |
| `tool_result` | call id, final result content, exit code, duration, truncation flag |
| `usage` | input/output/cache/reasoning token counts |
| `run_completed` | stop reason, latency ms, final assembled text |
| `run_failed` | error class (auth / rate-limit / timeout / provider error), message, retryable flag |

Model **version tagging (R6)**: `run_started` records both the *requested*
model string (e.g. `claude-sonnet-5`) and the *provider-resolved* identifier
returned in the response (e.g. dated snapshot), plus the endpoint's base URL —
so "same model name, different host" (e.g. OpenRouter vs. first-party) remains
distinguishable in analytics.

### 4.3 Credential handling

- Keys entered once in the Endpoint Manager; stored server-side encrypted with
  Fernet (key material from `LLM_ARENA_VAULT_KEY` env var or generated on first
  run into a `0600` file).
- API responses **never** include secrets — endpoints serialize with
  `has_credentials: true` + last-4 hint only.
- Request snapshots stored in the log are **scrubbed** of `Authorization` /
  `x-api-key` headers before persistence (log fidelity applies to content,
  never to secrets).

### 4.4 Tool Execution Runtime (R8)

Every run is an **agentic loop**: adapter streams model output → model emits
tool calls → executor runs them → results are appended to the run's message
history → the adapter continues the request — repeated until the model stops
without tool calls or a budget trips. No user interaction anywhere in the loop.

**Standard toolset** (defined once, translated into each provider's tool-def
format by the registry so every model gets functionally identical tools):

| Tool | Behavior |
|---|---|
| `bash` | Execute a shell command on the VM. Runs as the backend's user (full system access by design). `cwd` defaults to the run's workspace. Streams stdout/stderr as `tool_output_delta` events; captures exit code. Per-call timeout (default 120 s, model can request up to a cap). |
| `read_file` | Read any path on the VM (full read access). |
| `write_file` / `edit_file` | Write/edit files. Paths resolve relative to the run's workspace; absolute paths **outside** the workspace are allowed only when `allow_outside_workspace` is enabled on the endpoint config (default **off** — see write-root policy below). |
| `list_dir` | Directory listing. |
| `web_fetch` | HTTP GET with size cap (models on the VM can also just `curl` via bash; this exists as a cheaper structured path). |

**Workspace layout & write-root policy.** Each conversation gets a directory;
each run's model gets its own subdirectory beneath it:

```
data/conversations/{conversation_id}/
├── shared/                        # user-provided input files, visible to all models
├── claude-sonnet-5/               # {model-slug}: one workspace per endpoint+model
│   ├── .runs/{run_id}.manifest    # files created/modified per run (workspace diff)
│   └── … model's own artifacts …
├── gpt-5.2/
└── llama-4-local/
```

- Workspace slug = endpoint name + model, stable across turns of the same
  conversation, so a model's turn 5 can build on files it wrote in turn 2.
- Every model's system prompt is injected with a **workspace contract**: its
  absolute workspace path, an instruction that all artifacts must be created
  under it, and a note that `shared/` is read-only common input.
- Enforcement is layered: `read` is unrestricted (full VM access per R8);
  structured `write_file`/`edit_file` calls are *path-checked* against the
  workspace by default; `bash` is uncheckable by nature (root access is the
  requirement), so bash relies on the contract + cwd defaulting — and the
  **workspace manifest** (filesystem snapshot diff per tool call, recorded as
  part of `tool_result`) makes any out-of-workspace writes visible after the
  fact rather than silently lost.

**Budgets & safety valves** (per run, configurable per endpoint):
max agentic iterations (default 25), max total tool-execution wall time
(default 15 min), max output bytes per tool call (default 1 MiB, truncated
with marker), and a global **kill switch** (`POST /api/runs/{id}/cancel`
terminates the process group of any in-flight tool). Runaway detection: N
identical consecutive tool calls short-circuits with an error result.

**Concurrency model.** N models execute tools on the same VM simultaneously.
File-level collisions are prevented by per-model workspaces; *system-level*
collisions (apt/pip installs, port binding, killing processes) are not
preventable when everyone has root — mitigations: (a) system prompt contract
asks models to prefer virtualenvs and ephemeral ports, (b) an optional
**serialized-bash mode** per broadcast (global mutex on `bash`) for workloads
where interference is likely, (c) the deployment doc mandates a disposable VM
snapshot/restore workflow so a trashed machine is a reset, not an incident.

**Capture.** Every execution is recorded: the call args, cwd, start/end
timestamps, exit code, full stdout/stderr (up to cap, with truncation
markers), and the workspace manifest diff — all as `events` rows tagged to the
run, satisfying R7 for the tool dimension. Artifacts themselves stay on disk
in the workspace and are browsable/downloadable from the UI.

---

## 5. Data Model (the analytical record)

Guiding principle: **one normalized event table is the source of truth**;
everything else (conversations, runs, evaluations) is structure and indexing on
top of it. This is what makes "complete record … available for analytical
processing" real rather than aspirational.

```sql
-- Endpoint = a configured connection (provider + base_url + credentials + defaults)
endpoints(
  id UUID PK, name TEXT UNIQUE, provider_family TEXT,     -- 'anthropic'|'openai'|'openai_compat'|'google'
  base_url TEXT, default_model TEXT, default_params JSON,
  credential_ref TEXT,                                     -- vault handle, never the key
  enabled BOOL, created_at, updated_at
)

conversations(
  id UUID PK, title TEXT, system_prompt TEXT,              -- active system prompt (also logged per-run)
  created_at, updated_at, archived BOOL, metadata JSON     -- free-form tags for analytics
)

-- One user "send" that fans out to N endpoints
broadcasts(
  id UUID PK, conversation_id FK→conversations, user_message TEXT,
  system_prompt_snapshot TEXT,                             -- system prompt AS SENT, frozen
  created_at
)

-- One model's execution of one broadcast (also used for judge runs)
runs(
  id UUID PK, broadcast_id FK NULL, evaluation_id FK NULL, -- exactly one is set
  conversation_id FK,                                      -- denormalized for query speed
  endpoint_id FK→endpoints,
  model_requested TEXT, model_resolved TEXT,               -- name + version tags (R6)
  params JSON, request_snapshot JSON,                      -- full scrubbed outbound payload
  status TEXT,                                             -- pending|streaming|executing_tools|completed|failed|interrupted|cancelled
  workspace_path TEXT,                                     -- data/conversations/{conv}/{model-slug}
  tool_iterations INT,                                     -- agentic loop count
  stop_reason TEXT, latency_ms INT,
  input_tokens INT, output_tokens INT, reasoning_tokens INT,
  cache_read_tokens INT, cost_estimate_usd NUMERIC,
  started_at, completed_at
)

-- THE record: every atom of every exchange (R7)
events(
  id BIGINT PK AUTOINCREMENT,                              -- global total order
  run_id FK→runs NULL,                                     -- NULL for conversation-level events
  conversation_id FK, seq INT,                             -- per-run ordering
  event_type TEXT,   -- user_prompt|system_prompt|text_delta|text_final|thinking_delta|
                     -- thinking_final|tool_call|tool_result|usage|error|judge_verdict
  role TEXT,         -- user|assistant|system|tool|judge
  content JSON,      -- normalized payload
  raw JSON,          -- verbatim provider chunk/payload (nullable for user events)
  created_at TIMESTAMP
)

-- Structured record of every autonomous tool execution (R8) — the same data
-- also flows through `events`; this table is the analytics-friendly rollup
tool_executions(
  id UUID PK, run_id FK→runs, conversation_id FK,
  tool_name TEXT, call_id TEXT, arguments JSON,
  cwd TEXT, exit_code INT, duration_ms INT,
  stdout_bytes INT, stderr_bytes INT, truncated BOOL,
  files_created JSON, files_modified JSON,                 -- workspace manifest diff
  outside_workspace BOOL,                                  -- diff touched paths outside the run's dir
  started_at, completed_at
)

-- Judge workflow (R4/R5)
evaluations(
  id UUID PK, conversation_id FK, broadcast_id FK,
  judge_endpoint_id FK→endpoints, judge_model TEXT,
  candidate_run_ids JSON,                                  -- the selected responses
  rubric JSON,                                             -- dimensions + weights used
  status TEXT, created_at, completed_at
)

evaluation_results(       -- parsed, structured verdict (one row per candidate)
  id UUID PK, evaluation_id FK, candidate_run_id FK→runs,
  quality_score NUMERIC, accuracy_score NUMERIC, completeness_score NUMERIC,
  overall_score NUMERIC, rank INT,
  domains_capable JSON,   -- [{domain, evidence}]
  domains_failed JSON,    -- [{domain, evidence}]
  strengths TEXT, weaknesses TEXT, factual_issues JSON
)
```

Indexes: `events(conversation_id, id)`, `events(run_id, seq)`,
`runs(conversation_id)`, `runs(model_resolved)`, `evaluation_results(candidate_run_id)`.

**JSONL mirror:** the recorder also appends every `events` row to
`data/eventlog/YYYY-MM-DD.jsonl` (flat JSON, same fields). DuckDB one-liner
analytics with zero unload step: `SELECT … FROM read_json_auto('data/eventlog/*.jsonl')`.

**Blind-judging integrity note:** candidate responses are presented to the
judge as `Response A/B/C…` (anonymized, randomized order) to prevent
name-brand bias; the A/B/C→run_id mapping is stored on the evaluation row so
results are de-anonymized for display and analytics.

---

## 6. Backend API Surface

| Method & Path | Purpose |
|---|---|
| `GET/POST/PATCH/DELETE /api/endpoints` | Endpoint CRUD (R1) |
| `POST /api/endpoints/{id}/test` | Live health check + model listing |
| `GET /api/endpoints/{id}/models` | Enumerate models on that endpoint |
| `GET/POST /api/conversations` | Create/list conversations |
| `GET /api/conversations/{id}` | Full reconstructed transcript (all runs, all events) |
| `PATCH /api/conversations/{id}` | Title, system prompt, tags, archive |
| `POST /api/broadcasts` | `{conversation_id, message, endpoint_selections:[{endpoint_id, model, params}]}` → creates broadcast + N runs, returns ids immediately (R2) |
| `GET /api/broadcasts/{id}/stream` | **SSE**: multiplexed normalized events for all runs in the broadcast (R3) |
| `POST /api/runs/{id}/cancel` | Cancel one in-flight run (kills any executing tool's process group) |
| `GET /api/conversations/{id}/workspace?path=…` | Browse the conversation directory tree (shared + per-model subdirs) |
| `GET /api/conversations/{id}/workspace/file?path=…` | Download/preview a workspace artifact |
| `GET /api/runs/{id}/tools` | Structured tool-execution log for a run (`tool_executions` rows) |
| `POST /api/evaluations` | `{broadcast_id, candidate_run_ids, judge_endpoint_id, judge_model, rubric?}` (R4) |
| `GET /api/evaluations/{id}/stream` | SSE of the judge run (thinking + verdict as it forms) |
| `GET /api/evaluations/{id}` | Parsed structured verdict (R5) |
| `GET /api/analytics/models` | Aggregates per model: win rate, mean scores per dimension, domain matrix, latency, cost |
| `GET /api/analytics/export?conversation_id=…&format=jsonl\|csv` | Full-fidelity export (R7) |

### Broadcast orchestrator behavior

- `asyncio.TaskGroup`; one task per run; each task drives that run's full
  **agentic loop** (stream → execute tools → continue) with per-run budgets
  from §4.4 on top of a per-request timeout (default 300 s). Independent
  failure — one provider melting down or one model stuck in a tool loop never
  blocks the others' panels.
- Before dispatch, the orchestrator ensures the conversation directory and
  each selected model's workspace subdirectory exist, and injects the
  workspace contract into each model's system prompt.
- Conversation history sent to each model is that model's *own* prior turns
  in the conversation plus shared user turns (each endpoint sees a coherent
  bilateral conversation, not the other models' answers), assembled from the
  `events` table. A per-broadcast override ("fresh context") is a checkbox.
- Retry policy: automatic retry (max 2, exponential backoff) on 429/5xx
  *before* first token only; never mid-stream. Every attempt is logged.

---

## 7. Frontend Design

### Layout (single page, three zones)

```
┌─────────────┬──────────────────────────────────────────────────────┐
│ Sidebar     │  Response Grid (1 panel per selected endpoint)       │
│ ─────────   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│ Conversa-   │  │ ☑ Claude     │ │ ☑ GPT-5.2    │ │ ☐ Llama 4    │  │
│ tions list  │  │ Sonnet 5     │ │              │ │ (local)      │  │
│             │  │ ▸ thinking…  │ │ streamed txt │ │ streamed txt │  │
│ Endpoints   │  │ streamed txt │ │              │ │              │  │
│ manager     │  │ 🔧 tool call │ │ 1.2s · $0.004│ │ 3.8s · $0    │  │
│             │  └──────────────┘ └──────────────┘ └──────────────┘  │
│ Analytics   │  [☑☑ selected] → [⚖ Evaluate with… ▾]  → Verdict pane │
├─────────────┴──────────────────────────────────────────────────────┤
│ [system prompt ▾]  [endpoint selector: ●Claude ●GPT ○Llama +Add]    │
│ ┌──────────────────────────────────────────────────────┐ [Send ⏎]  │
│ │ chat composer…                                        │           │
│ └──────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### Response panel anatomy (R3, R7 visibility)

Each `ResponsePanel` renders, in stream order: collapsible **thinking block**
(distinct styling, collapsed by default once final text starts), **tool
execution cards** (tool name + syntax-highlighted args + a live mini-terminal
streaming stdout/stderr via `tool_output_delta`, then exit code, duration, and
files-touched chips), markdown final text, and a footer with latency / token
counts / cost estimate / tool-iteration count / model version chip / status.
Panel actions: select-for-evaluation checkbox, copy, re-run, cancel (kills
in-flight tools), expand to full screen, open workspace folder, view raw
request/response JSON.

A **Workspace tab** per conversation shows the directory tree
(`shared/` + one subdirectory per model) with file preview/download, so
artifacts the models produced are inspectable next to their transcripts —
and can be eyeballed when judging (see below).

Grid is responsive: 1–2 columns on narrow screens, up to 4 wide; panels are
individually scrollable with synchronized-scroll toggle for side-by-side
reading.

### Evaluation console (R4/R5)

1. Check ≥2 response panels → floating "Evaluate" bar appears.
2. Pick judge endpoint+model (any configured endpoint; sensible default
   remembered) and optionally edit the **rubric** (default dimensions:
   quality, accuracy, completeness, each 0–10 with definitions; user can add
   dimensions/weights — rubric JSON is stored on the evaluation).
3. Judge streams into a verdict pane: per-response scorecards, **ranking
   table**, pairwise comparison notes, and a **domain matrix** (rows =
   domains the judge identified in the task, e.g. "numerical reasoning",
   "API knowledge", columns = models, cells = capable/partial/failed with
   evidence quotes).
4. Judge prompting: structured-output request (JSON schema / tool-forced
   output) with the anonymized candidates (§5); the free-text reasoning is
   kept as the judge's logged thinking/response, the parsed JSON populates
   `evaluation_results`.
5. When candidate runs produced workspace artifacts, the judge gets read-only
   `read_file`/`list_dir` tools scoped to anonymized copies of the candidate
   workspaces, so it can verify that produced code/files actually work as
   claimed — not just grade the prose.

### State management

Zustand (small, no boilerplate): `endpointsStore`, `conversationStore`
(messages + runs + per-run event buffers), `evaluationStore`, plus a single
`useBroadcastStream(broadcastId)` hook wrapping `EventSource` that demuxes
events by `run_id` into the store. History replay uses the same reducers fed
from `GET /api/conversations/{id}` — streaming and replay share one code path.

---

## 8. Logging & Analytics (R6/R7 deep-dive)

What gets captured, per artifact class:

| Artifact | Where | Notes |
|---|---|---|
| User prompts | `events(event_type='user_prompt')` + `broadcasts.user_message` | |
| System prompts | `broadcasts.system_prompt_snapshot` + per-run `request_snapshot` | frozen as-sent, even if edited later |
| Full outbound request | `runs.request_snapshot` | scrubbed of credentials |
| Thinking blocks | `thinking_delta`/`thinking_final` events | raw provider form preserved in `raw` |
| Tool calls & results | `tool_call`/`tool_output_delta`/`tool_result` events + `tool_executions` table | args, full stdout/stderr, exit codes, durations, workspace file diffs |
| Workspace artifacts | `data/conversations/{conv}/{model-slug}/` on disk + per-run manifests | browsable in UI, included in export bundles |
| Streamed deltas | `text_delta` events | reconstructable typing timeline (inter-token latency analysis) |
| Final responses | `text_final` + `runs` terminal fields | |
| Errors/retries | `error` events + `runs.status` | error taxonomy for reliability analytics |
| Usage & cost | `usage` events + `runs` columns | costing table per model, versioned |
| Judge I/O | judge `runs` + `judge_verdict` events + `evaluation_results` | same fidelity as chat |
| Model identity | `runs.model_requested` + `model_resolved` + endpoint | R6 tagging |

Built-in analytics views (v1, read-only dashboards over SQL):

- **Model leaderboard:** mean scores per rubric dimension, win rate (rank 1
  share), evaluation count, filterable by date/tag.
- **Domain capability matrix:** aggregate of `domains_capable`/`domains_failed`
  across evaluations → heatmap of model × domain.
- **Ops view:** latency percentiles, failure rates, token/cost totals per
  endpoint.
- **Export:** JSONL/CSV per conversation or global; documented schema so
  pandas/DuckDB/notebooks can consume without reading source code.

---

## 9. Security & Privacy

**Threat model (changed by R8):** with autonomous tool execution, every
connected model effectively has shell access to the host. The application does
not pretend to sandbox this — **the VM is the security boundary**, and the
deployment requirements make that explicit:

- **Dedicated, disposable VM only.** Documented as a hard requirement: fresh
  VM or snapshot per project, nothing of value on the machine beyond the
  `data/` directory (which should be backed up/synced off-box), restore from
  snapshot when a model trashes the environment. Never run the backend on a
  workstation or shared server. A startup banner + config flag
  (`I_UNDERSTAND_TOOLS_HAVE_FULL_VM_ACCESS=1`) makes this consent explicit.
- **Credential hygiene matters more, not less.** Models with shell access can
  read the process environment and disk — so provider keys live only in the
  encrypted vault file, are decrypted in-memory per request, and are *not*
  exported into tool subprocess environments (executor spawns tools with a
  scrubbed env). This keeps one model from trivially harvesting the keys used
  to call its competitors. (A determined model with root can still find ways;
  the disposable-VM assumption is the real backstop — and `outside_workspace`
  / audit events make attempts visible.)
- **Egress awareness.** Tool-executing models can make arbitrary network
  calls. v1 documents this rather than restricting it (matching R8's "full
  control" intent); an optional egress allowlist at the VM firewall level is
  the recommended hardening for sensitive work.
- Credentials: encrypted at rest (§4.3), never serialized to the client,
  scrubbed from all logs/snapshots.
- The backend binds to `127.0.0.1` by default; a single shared bearer token
  (env-set) gates the API if bound wider. CORS locked to the frontend origin.
- Prompt/response content is sensitive by nature → the SQLite DB and JSONL
  logs live under `data/` (gitignored); README documents backup guidance.
- No telemetry; the only outbound app traffic is to user-configured endpoints
  (plus whatever the models themselves do via tools, which is logged).

---

## 10. Phased Delivery Plan

Each phase is independently shippable and ends with working software.

### Phase 0 — Scaffolding (small)
Directory layout, FastAPI app + health route, Vite app shell, SQLAlchemy +
Alembic wired, CI lint/test jobs, `llm-arena/README.md` quickstart.

### Phase 1 — Endpoint registry & vault (R1)
Endpoint CRUD API + vault; Anthropic and OpenAI/OpenAI-compatible adapters
with `list_models` + `health_check`; Endpoint Manager UI (add/edit/test/
enable). **Exit criteria:** add 3+ endpoints incl. a local Ollama, all pass
live test.

### Phase 2 — Broadcast chat & response grid (R2, R3)
Normalized event schema, `stream_chat` for both adapters (text + usage +
errors), broadcast orchestrator, SSE endpoint, Composer + EndpointSelector +
ResponseGrid with live streaming, cancel, re-run. **Exit criteria:** one
message fans out to 3 models streaming concurrently; one endpoint failing
doesn't disturb the others.

### Phase 3 — Full-fidelity recording (R6, R7)
Event Recorder (DB + JSONL), request snapshots, thinking-block capture
(Anthropic extended thinking; OpenAI-compatible `reasoning_content`),
conversation history reconstruction + replay UI, multi-turn context assembly.
**Exit criteria:** kill the server mid-stream → restart → transcript shows
accurate partial run marked `interrupted`; a DuckDB query over JSONL
reproduces a full conversation.

### Phase 3.5 — Tool Execution Runtime (R8)
Standard toolset + per-provider tool-def translation, executor with streaming
output and budgets, per-run workspaces + manifests + `tool_executions` table,
agentic loop in the orchestrator, tool execution cards + Workspace tab in the
UI, kill switch, serialized-bash mode, deployment/VM docs + consent flag.
**Exit criteria:** "write and run a script that does X" broadcast to 3 models
→ each autonomously creates and executes files in its own subdirectory, live
terminal output streams in each panel, manifests attribute every file to the
right model, and cancel mid-execution kills the process tree.

### Phase 4 — Judge & evaluation (R4, R5)
Rubric model + default rubric, anonymized judge prompt + structured-output
parsing, evaluation API + SSE, judge read-only access to anonymized candidate
workspaces, Evaluate bar, Verdict pane, RankingTable, DomainMatrix,
`evaluation_results` persistence. **Exit criteria:** select 3
responses → judged, ranked, domain-mapped verdict rendered and stored; judge
run itself fully logged.

### Phase 5 — Analytics, polish & hardening
Google Gemini adapter; leaderboard + domain heatmap + ops dashboards; JSONL/
CSV export endpoints; conversation search/tags; keyboard shortcuts; sync-
scroll; retry policy tuning; docs. **Exit criteria:** the analytics questions
in §8 answerable from the UI without SQL.

Rough sequencing: P0–P1 together first; P2 is the core-value milestone; P3
before P3.5 (recording must exist before autonomous execution — never run
unlogged tools) and both before P4 (the judge depends on faithful records and
artifacts); P5 iterative.

---

## 11. Testing Strategy

- **Adapters:** unit tests against recorded provider fixtures (streaming
  chunk sequences incl. thinking, tool calls, malformed/interleaved chunks,
  429s) — no live keys in CI. A `MockProvider` adapter (scripted responses,
  configurable delays/failures) is also a first-class *configurable endpoint*,
  which makes E2E tests and demos free.
- **Orchestrator:** concurrency tests — N runs, one hangs (timeout fires),
  one 429s (retry logged), cancellation mid-stream; assert DB state after each.
- **Recorder:** crash-recovery test (kill between deltas → `interrupted`);
  golden-file test that JSONL and DB agree.
- **Tool runtime:** executor tests — timeout kills process group, output
  truncation, runaway-loop short-circuit, workspace manifest diff accuracy,
  `outside_workspace` flagging, env scrubbing (spawned tool cannot see
  provider keys), concurrent runs writing to sibling workspaces without
  cross-talk, serialized-bash mutex ordering.
- **Judge:** verdict-parser tests over fixture judge outputs, incl.
  schema-violating output (graceful degradation: raw text stored, structured
  fields null, UI shows raw).
- **E2E:** Playwright against MockProvider endpoints — broadcast → stream →
  select → evaluate → verdict → reload page → history intact.

---

## 12. Risks & Open Questions

| Risk / question | Mitigation / proposed default |
|---|---|
| Provider streaming formats drift | raw payload always stored; adapter fixtures updated per SDK bump; normalized schema is additive-only |
| Thinking blocks unavailable/redacted on some models | capture what's offered, record `thinking_available: false` capability flag per run — absence is itself analytical data |
| Judge self-preference bias (judging its own family) | anonymized candidates (§5); analytics can slice results by judge model to expose bias |
| SQLite write contention under many concurrent streams | WAL mode + single writer task consuming an asyncio queue (recorder is already a single funnel) |
| Context window divergence in long multi-turn comparisons | per-model token accounting shown in panel footer; warn when a model's assembled history nears its limit |
| Tool calls auto-execute with full VM access (R8) | **Decided.** The VM is the security boundary (dedicated/disposable VM, consent flag, scrubbed subprocess env, egress guidance — §9); per-model workspaces + manifests provide filesystem attribution (§4.4) |
| One model damages the shared VM mid-comparison (kills the backend, fills the disk, breaks the toolchain for others) | Write-ahead recording preserves the record; per-run budgets bound the blast radius; serialized-bash mode for risky workloads; snapshot/restore is the documented recovery path |
| A model ignores the workspace contract via bash | Not preventable with root by design — detected instead: manifest diffs set `outside_workspace`, surfaced in UI and analytics (itself a useful "instruction-following" signal per model) |
| Multi-user later? | **Open** — schema has no user column yet; adding `owner_id` to `conversations`/`endpoints` is a small migration. Deferred deliberately. |
| Same repo or split out? | Start in `llm-arena/` here; it shares nothing with the audio detector at runtime, so extraction later is `git filter-repo`-trivial. |

---

## 13. Immediate Next Steps

1. Review/approve this plan (esp. §12 open questions and the SQLite default).
2. Phase 0 scaffold PR.
3. Phase 1: vault + Anthropic/OpenAI adapters + Endpoint Manager.
