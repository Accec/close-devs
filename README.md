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
export OPENAI_API_KEY=your_key
docker compose up -d postgres
aerich upgrade
python main.py run-once --config config/default.toml --repo .
```

If you want a no-key smoke path, set `provider = "mock"` in
`config/default.toml` before the first run.

## Remote Repositories

Close-Devs can also clone a remote repository directly into the report-local
runtime:

```bash
python main.py run-once \
  --config config/default.toml \
  --repo https://github.com/example/project.git \
  --repo-ref main
```

`--repo-ref` accepts a branch, tag, or commit SHA. Remote sources require
`[environment].enabled = true` because the runtime must materialize the
repository before scanning.

For private repositories you can use either HTTPS token auth or SSH key auth.

HTTPS token auth:

```bash
export GIT_AUTH_TOKEN=your_token
python main.py run-once \
  --config config/default.toml \
  --repo https://github.com/example/private-project.git \
  --repo-ref main
```

SSH key auth:

```bash
export GIT_SSH_KEY_PATH=/abs/path/to/id_ed25519
export GIT_KNOWN_HOSTS_PATH=/abs/path/to/known_hosts
python main.py run-once \
  --config config/default.toml \
  --repo git@github.com:example/private-project.git \
  --repo-ref main
```

The clone command is authenticated through environment variables, not by
embedding secrets into the command line, so tokens are not written into report
artifacts such as `install.log` or `environment.json`.

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

- `mock`: deterministic local test path
- `openai`: native OpenAI via `langchain_openai`
- `openai_compatible`: OpenAI-compatible gateways via `base_url`
- `anthropic`: native Anthropic via `langchain_anthropic`
- `google_genai`: native Gemini via `langchain_google_genai`
- `ollama`: local models via `langchain_ollama`

Provider routing is per-agent. `[llm]` defines the global default, and
`[agents.static]`, `[agents.dynamic]`, and `[agents.maintenance]` can override
`provider`, `model`, `base_url`, `api_key_env`, `temperature`,
`timeout_seconds`, and `system_prompt`.

Close-Devs now uses strict failure semantics for real providers. If a required
API key or provider dependency is missing, the run fails instead of silently
falling back to `mock`.

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
