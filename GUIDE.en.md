# Close-Devs Guide

## 1. What Close-Devs Is

Close-Devs is a Python 3.11+ repository maintenance system built around exactly
three core agents:

- `StaticReviewAgent`
- `DynamicDebugAgent`
- `MaintenanceAgent`

It is designed for long-term repository maintenance rather than one-off tasks.
The system continuously scans a repository, runs static and dynamic analysis,
generates safe maintenance patches, validates those patches, persists run
history, and publishes results locally or through GitHub PR workflows.

Close-Devs is not a "four-agent" design. The `Orchestrator` is a deterministic
supervisor and workflow runner, not a fourth autonomous agent.

## 2. Architecture and Runtime Model

The runtime model is:

- deterministic supervision
- autonomous agent sessions
- isolated per-run execution environments
- PostgreSQL-first persistence with SQLite compatibility

At a high level, a normal maintenance run looks like this:

1. Close-Devs scans the repository and computes the change set.
2. `StaticReviewAgent` and `DynamicDebugAgent` run in parallel.
3. Their findings and handoffs are merged into a maintenance task.
4. `MaintenanceAgent` decides what to inspect and produces a patch proposal.
5. Validation static and dynamic tasks run against the patched validation workspace.
6. Reports, findings, patches, skill metadata, and run history are persisted.

Each agent runs a multi-step session, not a single helper call. Within its
budget, an agent decides:

- what files to inspect
- which typed tools to call
- what evidence to keep
- when enough evidence exists to finalize

The current implementation uses:

- `asyncio` for async orchestration
- `LangGraph` for workflow graphs
- `Tortoise ORM` for persistence
- `Aerich` for database migrations

## 3. Three Agents and Their Boundaries

### `StaticReviewAgent`

Responsibilities:

- deterministic static tooling
- semantic code review
- architecture and correctness observations
- structured `finding` and `handoff` generation

It can:

- read files
- search the repository
- inspect diffs
- build AST summaries
- run static tools such as `ruff`, `mypy`, and `bandit`

It cannot:

- write files
- modify the repository
- run runtime mutation steps as if it were a debugger

### `DynamicDebugAgent`

Responsibilities:

- execute test and repro commands
- collect stderr/stdout and runtime evidence
- parse failures and tracebacks
- generate runtime-oriented fix requests

It can:

- run test commands
- parse traceback text
- inspect relevant files and logs
- iteratively refine diagnosis within its budget

It cannot:

- write files
- patch the repository
- act as the final static standards authority

### `MaintenanceAgent`

Responsibilities:

- consume upstream findings and handoffs
- inspect impacted files
- generate safe patch proposals
- prepare code changes inside the maintenance workspace

It can:

- read files
- inspect diffs
- prepare safe patches
- write files inside the maintenance workspace

It cannot:

- replace static review as the final static judgment
- replace dynamic debug as the final runtime diagnosis
- implicitly gain push/commit authority

Only `MaintenanceAgent` is allowed to write repository content, and even then it
writes to the report-local maintenance workspace by default. It does not write
back to the original repository unless `auto_apply_patch = true`.

## 4. Execution Environment and Isolated Runtime

Every run creates an isolated runtime under:

`reports/<run_id>/runtime/`

The layout is:

- `base_workspace/<repo_name>`
- `maintenance_workspace/<repo_name>`
- `validation_workspace/<repo_name>`
- `.venv`

Meaning:

- `base_workspace` is the initial analysis copy of the target repo
- `maintenance_workspace` is the writable workspace used by `MaintenanceAgent`
- `validation_workspace` is the copy used for validation after patching
- `.venv` is the report-local virtual environment used for analysis and validation

This matters because Close-Devs does not primarily trust the host environment.
Static checks, test runs, and validation are intended to run against the
report-local environment first.

### Dependency auto-detection

Dependency installation is currently auto-detected in this order:

1. `src/requirements.txt`
2. `requirements.txt`
3. `requirements-dev.txt`
4. `requirements-test.txt`
5. `pyproject.toml:project.dependencies`

If dependencies are found, Close-Devs will:

1. create a report-local venv
2. upgrade `pip`, `setuptools`, and `wheel`
3. install detected project dependencies
4. install analysis bootstrap tools such as `ruff`, `mypy`, `bandit`, and `pytest`

If installation fails, the run is marked `degraded`, but the workflow continues.
The degraded state is recorded in the report instead of silently falling back to
the host environment.

## 5. Installation and First Run

### PostgreSQL main path

Use this path if you want the default production-like local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export OPENAI_API_KEY=your_key
docker compose up -d postgres
aerich upgrade
python main.py run-once --config config/default.toml --repo .
```

If you want to test the runtime without a real provider, temporarily set
`[llm].provider = "mock"` or override a specific agent provider to `mock`.

### SQLite lightweight local path

Use this when you want fast local runs without Docker:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export OPENAI_API_KEY=your_key
export DATABASE_URL=sqlite:////tmp/close_devs_local.db
python main.py run-once --config config/default.toml --repo .
```

This works because Close-Devs reads `DATABASE_URL` first and can infer the
backend from the DSN.

### Remote repository path

Close-Devs can clone a remote Git repository directly into
`reports/<run_id>/runtime/` and run the whole workflow there:

```bash
python main.py run-once \
  --config config/default.toml \
  --repo https://github.com/example/project.git \
  --repo-ref main
```

`--repo-ref` accepts a branch, tag, or commit SHA. If `--repo` points to a
remote repository, `[environment].enabled` must remain `true`.

For private repositories, the default auth paths are:

- HTTPS token: `GIT_AUTH_TOKEN`
- SSH key: `GIT_SSH_KEY_PATH`
- optional known hosts file: `GIT_KNOWN_HOSTS_PATH`

HTTPS token example:

```bash
export GIT_AUTH_TOKEN=your_token
python main.py run-once \
  --config config/default.toml \
  --repo https://github.com/example/private-project.git \
  --repo-ref main
```

SSH example:

```bash
export GIT_SSH_KEY_PATH=/abs/path/to/id_ed25519
export GIT_KNOWN_HOSTS_PATH=/abs/path/to/known_hosts
python main.py run-once \
  --config config/default.toml \
  --repo git@github.com:example/private-project.git \
  --repo-ref main
```

The runtime injects auth through environment variables rather than placing
secrets into the clone command string, so tokens are not written into
`install.log` or `environment.json`.

### First-run expectations

A successful first run should:

- create a database run record
- create `reports/<run_id>/`
- create `reports/<run_id>/runtime/.venv`
- produce `summary.md`, `report.json`, `findings.json`
- optionally produce `patch.diff`

## 6. Configuration Reference

The primary runtime configuration lives in:

- `config/default.toml`

### `[app]`

Controls:

- target repo root
- state directory
- reports directory
- scan interval
- log level
- whether agent activity logging is enabled
- whether generated patches are auto-applied to the source repo
- include and exclude patterns

Important fields:

- `repo_root`
- `reports_dir`
- `log_level`
- `log_agent_activity`
- `auto_apply_patch`
- `rules_path`
- `include`
- `exclude`

### `[llm]`

Controls the model provider used by all agent runtimes unless overridden by
per-agent config.

Important fields:

- `provider = "mock" | "openai" | "openai_compatible" | "anthropic" | "google_genai" | "ollama"`
- `model`
- `base_url`
- `api_key_env`
- `timeout_seconds`
- `temperature`
- `max_retries`
- `system_prompt`

Provider-specific defaults:

- `openai` -> `OPENAI_API_KEY`
- `openai_compatible` -> `OPENAI_API_KEY`
- `anthropic` -> `ANTHROPIC_API_KEY`
- `google_genai` -> `GOOGLE_API_KEY`
- `ollama` -> default `base_url = "http://127.0.0.1:11434"` and no API key

If a real provider is selected and the required credentials are missing,
Close-Devs fails fast. It does not silently fall back to `mock`.

### `[static_review]`

Controls static tool behavior:

- `max_complexity`
- `ruff_command`
- `mypy_command`
- `bandit_command`

### `[dynamic_debug]`

Controls runtime command behavior:

- `smoke_commands`
- `test_commands`
- `timeout_seconds`

### `[database]`

Controls persistence backend:

- `backend = "postgres" | "sqlite"`
- `url`
- `url_env`
- `echo`

`DATABASE_URL` is the normal override path for local development, CI, and
one-off debugging.

### `[environment]`

Controls report-local isolated runtime creation:

- `enabled`
- `scope`
- `install_mode`
- `install_fail_policy`
- `python_executable`
- `bootstrap_tools`
- `git_auth_mode`
- `git_https_token_env`
- `git_https_username`
- `git_ssh_key_path`
- `git_ssh_key_path_env`
- `git_known_hosts_path`
- `git_known_hosts_path_env`
- `git_ssh_strict_host_key_checking`
- `git_clone_timeout_seconds`

Current defaults mean:

- environment isolation is enabled
- it applies to all analysis
- dependency installation is auto-detected
- failures mark the run as degraded instead of failing fast
- remote repository clone auth defaults to auto-detection

### `[skills]`

Controls the repo skill system and shadow self-upgrade:

- `enabled`
- `repo_root`
- `shadow_evaluation_enabled`
- `min_shadow_runs`
- `promotion_margin`

### `[skills.static]`, `[skills.dynamic]`, `[skills.maintenance]`

Each agent skill section currently controls:

- `baseline`
- `auto_upgrade`

### `[agents.static]`, `[agents.dynamic]`, `[agents.maintenance]`

These sections define hard runtime ceilings and tool permissions for each
agent.

Important fields:

- `model`
- `max_steps`
- `max_tool_calls`
- `max_wall_time_seconds`
- `max_consecutive_failures`
- `max_budget_ceiling`
- `safety_lock`
- `allowed_tools`
- `allowed_tool_superset`

Important rule:

- skills can optimize within these bounds
- skills cannot break these bounds

### `[github]`

Controls PR workflow publishing behavior:

- provider and repo context
- token env var
- fix branch prefix
- review mode
- artifact retention

### `[pr_workflow]`

Controls PR publishing policy:

- inline comment limit
- whether companion PRs are allowed
- safe-fix-only mode
- issue comment rerun trigger

## 7. Local CLI Workflows

All public local entrypoints currently come from `main.py`.

### `run-once`

```bash
python main.py run-once --config config/default.toml --repo .
```

Use when:

- you want the full maintenance loop
- you want static, dynamic, maintenance, and validation in one run

Main effects:

- creates a run record
- creates a report-local environment
- runs the full graph
- writes a report directory

### `scan`

```bash
python main.py scan --config config/default.toml --repo .
```

Use when:

- you only want repository scan counts
- you want to verify include/exclude behavior

Main output:

- tracked file count
- changed file count
- added/removed file count

### `review`

```bash
python main.py review --config config/default.toml --repo .
```

Use when:

- you only want static review behavior
- you are tuning static rules or static skills

Main output:

- a static-review-only report
- findings and skill metadata

### `debug`

```bash
python main.py debug --config config/default.toml --repo .
```

Use when:

- you only want runtime diagnosis
- you are tuning repro commands or dynamic skills

Main output:

- a dynamic-debug-only report

### `report`

```bash
python main.py report --config config/default.toml --repo .
python main.py report --config config/default.toml --repo . --show
```

Use when:

- you want the latest report directory
- you want to print the latest markdown summary to the terminal

### `skill-status`

```bash
python main.py skill-status --config config/default.toml --repo .
python main.py skill-status --config config/default.toml --repo . --agent static_review
```

Use when:

- you want to see the active skill version for each agent
- you want to know whether a candidate exists
- you want to see whether a binding is frozen

Main output includes:

- active version
- source
- candidate version
- shadow run count
- candidate status
- frozen state

### `skill-history`

```bash
python main.py skill-history --config config/default.toml --repo . --agent static_review
```

Use when:

- you want recent evaluation records for one agent

Main output includes:

- run id
- active version
- candidate version
- active score
- candidate score
- whether promotion happened
- recorded reasons

### `skill-freeze`

```bash
python main.py skill-freeze --config config/default.toml --repo . --agent maintenance
python main.py skill-freeze --config config/default.toml --repo . --agent maintenance --unfreeze
```

Use when:

- you want to stop automatic promotion for an agent
- you want to re-enable automatic promotion later

### `skill-promote`

```bash
python main.py skill-promote --config config/default.toml --repo . --agent dynamic_debug
```

Use when:

- you want to manually promote the current open candidate

Effect:

- updates the DB binding pointer if an open candidate exists
- does not rewrite the repo skill pack files

## 8. GitHub PR Workflows

Close-Devs uses a two-phase PR workflow.

### Phase 1: `pr-review`

```bash
python main.py pr-review \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --repo "$GITHUB_WORKSPACE"
```

Use when:

- a GitHub Actions job needs to analyze a PR
- you want a report and publish context, but not publishing yet

Main effects:

- loads PR context
- scans PR repo state
- runs static and dynamic agents
- runs maintenance and validation
- writes report artifacts
- writes `artifacts/publish_context.json`

### Phase 2: `pr-publish`

```bash
python main.py pr-publish \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --publish-context "reports/<run_id>/artifacts/publish_context.json" \
  --repo "$GITHUB_WORKSPACE"
```

Use when:

- the report artifact has already been uploaded
- you want to publish summary comment, artifact links, inline comments, and
  optional companion PR output

Main effects:

- resolves artifact URLs
- updates one stable Close-Devs summary comment
- optionally publishes conservative inline comments
- optionally opens or updates a companion PR

### PR publish modes

Current publish modes are:

- `companion_pr`
- `comment_only`
- `artifact_only`

Close-Devs degrades automatically when:

- the token is missing
- permissions are insufficient
- the PR comes from a fork
- validation makes the patch non-publishable

## 9. Skill System and Shadow Self-Upgrade

The repo skill system is one of the main runtime controls for agent behavior.

Baseline skill packs live in:

- `config/skills/static/`
- `config/skills/dynamic/`
- `config/skills/maintenance/`

Each pack currently contains:

- `manifest.toml`
- `policy.toml`
- `skill.md`
- `examples.json`

### What a skill pack controls

A skill pack defines behavior such as:

- prompt guidance
- planning heuristics
- tool preference
- severity and prioritization bias
- handoff style
- completion checklist
- reflection and upgrade hint style

### Active skill vs candidate skill

- The repo baseline skill is the versioned source-of-truth template in git.
- The active skill is the version currently bound in the database.
- A candidate skill is a proposed next version stored in the database.

The active binding can override the baseline selection without rewriting repo
files.

### Shadow evaluation

Shadow evaluation means:

- production runs still use the active skill
- candidate skills are evaluated alongside active behavior using deterministic
  scoring
- promotion only happens after enough shadow runs and sufficient improvement

Current promotion controls come from `[skills]`:

- `min_shadow_runs`
- `promotion_margin`

### Safety boundaries

Skills cannot:

- give `StaticReviewAgent` write access
- give `DynamicDebugAgent` write access
- give `MaintenanceAgent` default push/commit authority
- break hard ceilings defined in `[agents.*]`

Skills can:

- change preferred tool order
- change prioritization
- change handoff style
- recommend smaller budgets within the hard ceiling

### How to operate the skill system

Use:

- `skill-status` to inspect current state
- `skill-history` to review evaluation records
- `skill-freeze` to stop automatic promotion
- `skill-promote` to manually activate an open candidate

## 10. Reports, Artifacts, and How to Read Them

Each run writes:

- `reports/<run_id>/summary.md`
- `reports/<run_id>/report.json`
- `reports/<run_id>/findings.json`
- `reports/<run_id>/patch.diff`
- `reports/<run_id>/artifacts/`

### Main report artifacts

#### `summary.md`

Human-readable report overview. It currently includes:

- workflow metadata
- execution environment status
- agent skills summary
- static review result
- dynamic debug result
- maintenance result
- validation results

#### `report.json`

Machine-readable full workflow report, including metadata and agent result
artifacts.

#### `findings.json`

Flattened findings across the run.

#### `patch.diff`

Unified diff for the maintenance patch proposal when a patch exists.

#### `artifacts/environment.json`

Machine-readable summary of the isolated runtime:

- runtime paths
- dependency sources
- install commands
- install failures
- whether the environment is degraded

#### `artifacts/install.log`

Raw install transcript for venv/bootstrap/dependency installation.

#### PR-only artifacts

PR workflows may also produce:

- `artifacts/publish_context.json`
- `artifacts/review_payload.json`

### How to read workflow status correctly

Important interpretation rule:

- `Status: succeeded` means the workflow executed successfully
- it does **not** automatically mean the repository has been fixed

You must also inspect:

- static findings
- dynamic findings
- maintenance patch output
- validation results

### How to tell whether a problem is unresolved

A run likely did not fix the main issue when:

- `dynamic_validation` still reports the original blocker
- `maintenance` produced only low-risk cosmetic patches
- `validation` still contains high-severity findings
- the report mentions `regressed` or unresolved comparisons

### How to tell whether real AI ran

Look for these signals:

- startup logs show `provider=...` and `client=...` for each agent
- `summary.md` or `report.json` records `actual_llm_provider` / `actual_llm_providers`
- summaries and handoffs contain model-driven reasoning rather than only
  deterministic tool output

If the run is using `mock`, runtime metadata will say so explicitly.

## 11. Logging and Agent Activity

Close-Devs can log agent activity in detail when:

- `[app].log_agent_activity = true`

Common log events:

- `Task dispatched`
- `Session started`
- `Step decided`
- `Tool started`
- `Tool finished`
- `Session finalizing`
- `Session finished`
- `Task finished`

These logs help you answer:

- which agent ran
- which model client it used
- which tools it called
- how many steps it took
- whether it hit budget or finished normally

### Interpreting agent logs

Example interpretation:

- `Task dispatched`: the supervisor handed a task to an agent
- `Session started`: the autonomous session is live with a specific budget and toolset
- `Step decided`: the agent chose its next action
- `Tool started`: the selected tool call began
- `Tool finished`: the tool returned with success or failure
- `Session finished`: the agent completed, exhausted budget, or errored

## 12. Database and Migrations

Close-Devs is PostgreSQL-first, with SQLite compatibility.

### PostgreSQL path

Use PostgreSQL when:

- running locally in the default configuration
- running CI
- running PR workflows
- keeping richer long-term history

Setup:

```bash
docker compose up -d postgres
aerich upgrade
```

### SQLite path

Use SQLite when:

- you want fast local smoke tests
- you do not want to run Docker
- you want disposable single-user runs

Example:

```bash
export DATABASE_URL=sqlite:////tmp/close_devs_local.db
python main.py run-once --config config/default.toml --repo .
```

### Migration rule

If the database exists but the tables do not, run:

```bash
aerich upgrade
```

## 13. Troubleshooting

### PostgreSQL connection refused

Symptoms:

- startup fails while connecting to `127.0.0.1:5432`

Fix:

```bash
docker compose up -d postgres
aerich upgrade
```

Also check whether `DATABASE_URL` is overriding the default DSN.

### `aerich upgrade` has not been run

Symptoms:

- errors mentioning missing tables such as `runs` or `aerich`

Fix:

```bash
aerich upgrade
```

### Missing `OPENAI_API_KEY`

Symptoms:

- startup fails before the workflow begins
- the error says a required API key env var is missing
- `provider = "openai"` or `provider = "openai_compatible"` is configured

Fix:

```bash
export OPENAI_API_KEY=your_key
```

Then rerun the command.

For other providers, set the matching env var:

- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

### Report-local environment is degraded

Symptoms:

- `Execution Environment` says `degraded`
- `install_failures` is non-zero
- tests or tools fail due to missing packages

What to inspect:

- `artifacts/environment.json`
- `artifacts/install.log`

Typical causes:

- bad dependency file
- private dependency access failure
- unsupported dependency layout

### `pr-publish` cannot find artifact or cannot update comments

Symptoms:

- publish phase cannot resolve artifact URLs
- PR summary comment is not updated

Check:

- GitHub Actions uploaded the run artifact
- `GITHUB_TOKEN` is present
- the token has comment/publish permissions
- the `publish_context.json` path is correct

### SQLite says `database is locked`

Symptoms:

- local SQLite runs fail with `database is locked`

Common reason:

- multiple processes are using the same SQLite file concurrently

Fix:

- use a separate SQLite file per smoke run
- or stop concurrent local runs
- or use PostgreSQL for repeated concurrent runs

## 14. Recommended Operating Practices

For operators:

- use PostgreSQL by default
- keep `log_agent_activity = true`
- inspect `summary.md` before trusting `Status: succeeded`
- check validation results before accepting a patch

For contributors:

- use SQLite for quick local smoke runs when convenient
- use `skill-status` and `skill-history` when tuning agent behavior
- freeze skill promotion if you are debugging a specific regression
- treat repo skill packs as versioned baselines, not ephemeral runtime state

For repository maintenance:

- keep dependency files accurate so the isolated runtime can bootstrap itself
- prefer safe autofix defaults until validation is consistently strong
- treat agent summaries as evidence, but rely on report artifacts and validation
  results for final judgment
