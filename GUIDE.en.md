# Close-Devs Guide

## Overview

Close-Devs is a Python 3.11+ repository maintenance system built around exactly
three core agents:

- `MaintenanceAgent`
- `StaticReviewAgent`
- `DynamicDebugAgent`

It uses `asyncio`, `LangGraph`, and `Tortoise ORM` to scan a repository,
analyze code, generate safe maintenance patches, validate those patches, and
persist run history.

## Project Layout

The source tree is flattened under `src/`:

- `src/agents`: the three core agents
- `src/core`: orchestration, CLI, models, dispatcher, scheduler
- `src/tools`: reusable utilities such as patching, command execution, static tooling
- `src/workflows`: workflow entrypoints
- `src/memory`: database and run history persistence
- `src/github`: GitHub PR publishing and rendering
- `src/repo`: repository scanning and change detection
- `src/reports`: report serialization and markdown rendering

Entry point:

- `main.py`

Default configuration:

- `config/default.toml`

GitHub Actions workflow:

- `.github/workflows/close-devs-pr.yml`

## Requirements

- Python `3.11+`
- Git
- Docker, if you want the default PostgreSQL runtime locally

Optional but recommended external tools:

- `ruff`
- `mypy`
- `bandit`
- `pytest`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d postgres
aerich upgrade
python main.py run-once --config config/default.toml --repo .
```

## CLI Commands

### Local maintenance loop

```bash
python main.py run-once --config config/default.toml --repo .
```

Runs the full maintenance workflow:

1. scan repository
2. run static review
3. run dynamic debug
4. generate maintenance patch proposal
5. validate the patch
6. write reports

### Repository scan only

```bash
python main.py scan --config config/default.toml --repo .
```

### Static review only

```bash
python main.py review --config config/default.toml --repo .
```

### Dynamic debug only

```bash
python main.py debug --config config/default.toml --repo .
```

### Show latest report

```bash
python main.py report --config config/default.toml --repo .
python main.py report --config config/default.toml --repo . --show
```

## GitHub PR Workflow

Close-Devs uses a two-phase PR workflow.

### Phase 1: analyze PR

```bash
python main.py pr-review \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --repo "$GITHUB_WORKSPACE"
```

This phase:

- loads PR context
- scans changed files
- runs static and dynamic analysis
- builds a maintenance patch proposal
- validates the patch
- writes `report.json` and `artifacts/publish_context.json`

### Phase 2: publish PR results

```bash
python main.py pr-publish \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --publish-context "reports/<run_id>/artifacts/publish_context.json" \
  --repo "$GITHUB_WORKSPACE"
```

This phase:

- resolves GitHub Actions artifact URLs
- updates one stable Close-Devs top-level PR comment
- publishes conservative inline comments when eligible
- opens or updates a companion PR for safe autofixes when publishable

## Reports and Artifacts

Each run writes a directory under `reports/<run_id>/`:

- `summary.md`
- `findings.json`
- `report.json`
- `patch.diff`
- `artifacts/`

For PR workflows, `artifacts/` also includes:

- `publish_context.json`
- `review_payload.json` after `pr-publish`

## Configuration

The main sections in `config/default.toml` are:

- `[app]`: repo root, report paths, include/exclude rules
- `[llm]`: `mock` or `openai_compatible`
- `[static_review]`: commands for `ruff`, `mypy`, `bandit`
- `[dynamic_debug]`: smoke and test commands
- `[github]`: branch prefix, token env, review mode, artifact retention
- `[pr_workflow]`: inline comment limit, companion PR enablement, safe-fix behavior
- `[database]`: backend and DSN

Important default behavior:

- default database backend: PostgreSQL
- environment variable `DATABASE_URL` overrides the DSN in config
- `auto_apply_patch = false` by default
- local LLM mode defaults to `mock`

## Database Notes

Default path:

- PostgreSQL via Docker Compose

Compatibility path:

- SQLite for lightweight local development or test runs

Migrations:

```bash
aerich upgrade
```

## What the Three Agents Do

### `StaticReviewAgent`

- static analysis only
- no code execution
- no patch writing

### `DynamicDebugAgent`

- runtime execution, tests, logs, tracebacks
- no static governance decision
- no patch writing

### `MaintenanceAgent`

- patch generation and safe autofix only
- no final static review judgment
- no runtime diagnosis

## Common Local Workflow

When developing Close-Devs itself:

1. run `python main.py run-once --config config/default.toml --repo .`
2. inspect the latest report
3. run `pytest`
4. adjust config or workflow behavior
5. rerun

## Troubleshooting

### PostgreSQL connection refused

- make sure Docker is running
- start PostgreSQL with `docker compose up -d postgres`
- verify `DATABASE_URL`

### `pr-publish` cannot find artifacts

- make sure `pr-review` ran first
- make sure the workflow uploaded the report directory
- verify the `publish_context.json` path

### No comment posted back to GitHub

- verify `GITHUB_TOKEN`
- check whether the PR comes from a fork
- inspect the workflow logs for capability degradation to `artifact_only`

## Recommended Reading Order

1. `README.md`
2. `GUIDE.en.md` or `GUIDE.zh-CN.md`
3. `config/default.toml`
4. `src/core/orchestrator.py`
5. `.github/workflows/close-devs-pr.yml`
