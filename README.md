# Close-Devs

Close-Devs is a Python 3.11+ MVP for long-term repository maintenance with
exactly three core agents:

- `MaintenanceAgent`
- `StaticReviewAgent`
- `DynamicDebugAgent`

The system uses `asyncio` and `LangGraph` to scan a repository, detect
incremental changes, run static and dynamic analysis concurrently, generate
maintenance patches, validate those patches in an isolated workspace, and
persist the full run history in PostgreSQL via Tortoise ORM.

Each of the three agents now runs as an autonomous session rather than a
single-shot helper:

- `StaticReviewAgent` explores code, tools, and local context within a read-only tool budget
- `DynamicDebugAgent` iterates on repro commands, test runs, and traceback analysis within a read-only execution budget
- `MaintenanceAgent` consumes upstream handoffs and is the only agent allowed to write repository files

The supervisor remains deterministic, but the work inside each agent is now
session-based: tool calls, handoffs, completion reasons, and agent summaries are
persisted as first-class runtime records.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d postgres
aerich upgrade
python main.py run-once --config config/default.toml --repo .
```

## Further Reading

For day-to-day operation, configuration, report interpretation, skill
management, and troubleshooting, use the full guides:

- English: [GUIDE.en.md](GUIDE.en.md)
- 中文: [GUIDE.zh-CN.md](GUIDE.zh-CN.md)

Use the guides if you are:

- operating Close-Devs against a real repository
- tuning agent behavior, skills, or runtime budgets
- debugging isolated environments, reports, PR publishing, or database setup

This README stays intentionally short and focuses on project overview and quick
start.

## GitHub PR Workflow

Close-Devs also supports an async pull-request maintenance loop for GitHub
Actions:

```bash
python main.py pr-review \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --repo "$GITHUB_WORKSPACE"

python main.py pr-publish \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --publish-context "reports/<run_id>/artifacts/publish_context.json" \
  --repo "$GITHUB_WORKSPACE"
```

The PR workflow keeps the three-agent boundary intact:

- `StaticReviewAgent` inspects changed files and rules only
- `DynamicDebugAgent` executes tests and repro commands only
- `MaintenanceAgent` generates safe autofix patches only

For same-repo PRs with publishable safe fixes, Close-Devs opens or updates a
companion PR from `close-devs/fix/<pr-number>/<run-id>`. Fork PRs or
restricted tokens automatically degrade to comment-only or artifact-only mode.
The default GitHub Actions entrypoint is defined in
`.github/workflows/close-devs-pr.yml`.

PR publishing is two-phase by default:

- `pr-review` runs analysis only and writes `report.json` plus `artifacts/publish_context.json`
- the workflow uploads the report directory as a GitHub Actions artifact
- `pr-publish` resolves the real artifact URL, updates one stable Close-Devs PR comment, optionally emits conservative inline comments, and opens or updates the companion PR when safe autofixes are publishable

## Design Principles

- Three core agents only
- Deterministic supervision with autonomous agent sessions
- Async orchestration with LangGraph
- Standard library first
- Patch proposal mode by default
- Long-term repository memory via PostgreSQL, SQLite compatibility, and report artifacts
- Strict write boundary: only `MaintenanceAgent` can modify files

## Agentic Runtime

Close-Devs now uses a shared `AgentKernel` for all three agents. The kernel
handles:

- multi-step session loops
- typed tool invocation
- per-agent budgets
- permission checks
- completion reasons
- session persistence

Per-agent runtime limits and allowed toolsets are configured in
`config/default.toml`:

- `[agents.static]`
- `[agents.dynamic]`
- `[agents.maintenance]`

Each section controls `model`, `max_steps`, `max_tool_calls`,
`max_wall_time_seconds`, `max_consecutive_failures`, and `allowed_tools`.

## LLM Providers

- `mock`: deterministic local fallback
- `openai_compatible`: any OpenAI-compatible endpoint via `base_url`, `model`,
  and an API key env var configured in `config/default.toml`

## Reports

Each run writes:

- `reports/<run_id>/summary.md`
- `reports/<run_id>/findings.json`
- `reports/<run_id>/report.json`
- `reports/<run_id>/patch.diff`
- `reports/<run_id>/artifacts/`

`summary.md` also includes agent session summaries such as step count, tool call
count, completion reason, and handoff count.

## Database Backends

- Default runtime backend: PostgreSQL
- Compatibility backend: SQLite for fast local tests and lightweight development
- Migration tool: Aerich

The default config reads `DATABASE_URL` first and falls back to the DSN in
`config/default.toml`.
