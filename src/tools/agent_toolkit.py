from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import tomllib
from typing import Any, Awaitable, Callable

from core.models import (
    AgentKind,
    AgentSession,
    EvidenceArtifact,
    FeedbackBundle,
    Finding,
    FixRequest,
    SafeFixPolicy,
    Severity,
    StartupConfigAnchor,
    StartupEntrypoint,
    StartupTopology,
    ToolPermissionSet,
    ToolResult,
    ToolSpec,
)
from tools.command_runner import CommandRunner
from tools.file_store import FileStore
from tools.language_support import detect_snippet_language
from tools.patch_service import PatchService
from tools.static_tooling import StaticTooling
from tools.test_runner import TestRunner
from tools.traceback_parser import TracebackParser


ToolHandler = Callable[[AgentSession, dict[str, Any]], Awaitable[ToolResult]]


@dataclass(slots=True)
class AgentTool:
    spec: ToolSpec
    handler: ToolHandler

    async def invoke(self, session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
        return await self.handler(session, tool_input)


class AgentToolkitFactory:
    def __init__(
        self,
        *,
        file_store: FileStore | None = None,
        command_runner: CommandRunner | None = None,
        static_tooling: StaticTooling | None = None,
        test_runner: TestRunner | None = None,
        traceback_parser: TracebackParser | None = None,
        patch_service: PatchService | None = None,
        safe_fix_policy: SafeFixPolicy | None = None,
    ) -> None:
        self.file_store = file_store or FileStore()
        self.command_runner = command_runner or CommandRunner()
        self.static_tooling = static_tooling or StaticTooling(self.command_runner)
        self.test_runner = test_runner or TestRunner(self.command_runner)
        self.traceback_parser = traceback_parser or TracebackParser()
        self.patch_service = patch_service or PatchService(self.file_store)
        self.safe_fix_policy = safe_fix_policy or SafeFixPolicy()

    def build_static_toolkit(
        self,
        permissions: ToolPermissionSet,
        *,
        context: "RunContext",
    ) -> dict[str, AgentTool]:
        return self._filter_tools(
            permissions,
            {
                "read_file": self._tool_read_file(),
                "search_repo": self._tool_search_repo(context),
                "git_diff": self._tool_git_diff(context),
                "ast_summary": self._tool_ast_summary(),
                "run_static_review": self._tool_run_static_review(context),
                "shell_readonly": self._tool_shell_readonly(context),
            },
        )

    def build_dynamic_toolkit(
        self,
        permissions: ToolPermissionSet,
        *,
        context: "RunContext",
    ) -> dict[str, AgentTool]:
        return self._filter_tools(
            permissions,
            {
                "read_file": self._tool_read_file(),
                "search_repo": self._tool_search_repo(context),
                "run_test_command": self._tool_run_test_command(context),
                "parse_traceback": self._tool_parse_traceback(),
                "shell_readonly": self._tool_shell_readonly(context),
            },
        )

    def build_maintenance_toolkit(
        self,
        permissions: ToolPermissionSet,
        *,
        context: "RunContext",
    ) -> dict[str, AgentTool]:
        return self._filter_tools(
            permissions,
            {
                "read_file": self._tool_read_file(),
                "search_repo": self._tool_search_repo(context),
                "git_diff": self._tool_git_diff(context),
                "prepare_safe_patch": self._tool_prepare_safe_patch(context),
                "write_file": self._tool_write_file(),
            },
        )

    async def discover_startup_topology(self, repo_root: Path) -> StartupTopology:
        topology = StartupTopology(
            repo_root=str(repo_root),
            src_layout=repo_root.joinpath("src").is_dir(),
        )

        entrypoints: list[StartupEntrypoint] = []
        anchors: list[StartupConfigAnchor] = []

        def register_entrypoint(
            path: str,
            *,
            context: str,
            config_anchor_path: str | None,
            repair_hint: str | None,
            notes: list[str] | None = None,
        ) -> None:
            if any(item.path == path and item.context == context for item in entrypoints):
                return
            entrypoints.append(
                StartupEntrypoint(
                    path=path,
                    context=context,
                    config_anchor_path=config_anchor_path,
                    repair_hint=repair_hint,
                    notes=list(notes or []),
                )
            )

        def register_anchor(
            path: str,
            *,
            context: str,
            anchor_type: str,
            status: str,
            repair_hint: str | None,
        ) -> None:
            if any(
                item.path == path and item.context == context and item.anchor_type == anchor_type
                for item in anchors
            ):
                return
            anchors.append(
                StartupConfigAnchor(
                    path=path,
                    context=context,
                    anchor_type=anchor_type,
                    status=status,
                    repair_hint=repair_hint,
                )
            )

        env_target = self._select_env_target(repo_root)
        register_anchor(
            env_target,
            context="config",
            anchor_type="env_template",
            status="present" if (repo_root / env_target).exists() else "missing",
            repair_hint="env-template-placeholder",
        )

        if (repo_root / "manage.py").is_file():
            status = await self._python_bootstrap_status(repo_root, Path("manage.py"))
            register_anchor(
                "manage.py",
                context="django_manage",
                anchor_type="python_src_bootstrap",
                status=status,
                repair_hint="managepy-sys-path-src",
            )
            register_entrypoint(
                "manage.py",
                context="django_manage",
                config_anchor_path="manage.py",
                repair_hint="managepy-sys-path-src",
            )

        if (repo_root / "asgi.py").is_file():
            status = await self._python_bootstrap_status(repo_root, Path("asgi.py"))
            register_anchor(
                "asgi.py",
                context="asgi",
                anchor_type="python_src_bootstrap",
                status=status,
                repair_hint="asgi-entrypoint-sys-path-src",
            )
            register_entrypoint(
                "asgi.py",
                context="asgi",
                config_anchor_path="asgi.py",
                repair_hint="asgi-entrypoint-sys-path-src",
            )

        if (repo_root / "wsgi.py").is_file():
            status = await self._python_bootstrap_status(repo_root, Path("wsgi.py"))
            register_anchor(
                "wsgi.py",
                context="wsgi",
                anchor_type="python_src_bootstrap",
                status=status,
                repair_hint="wsgi-entrypoint-sys-path-src",
            )
            register_entrypoint(
                "wsgi.py",
                context="wsgi",
                config_anchor_path="wsgi.py",
                repair_hint="wsgi-entrypoint-sys-path-src",
            )

        uvicorn_runner = self._select_uvicorn_runner_target(repo_root)
        if uvicorn_runner is not None:
            status = await self._python_bootstrap_status(repo_root, uvicorn_runner)
            register_anchor(
                uvicorn_runner.as_posix(),
                context="uvicorn",
                anchor_type="python_src_bootstrap",
                status=status,
                repair_hint="uvicorn-runner-sys-path-src",
            )
            register_entrypoint(
                uvicorn_runner.as_posix(),
                context="uvicorn",
                config_anchor_path=uvicorn_runner.as_posix(),
                repair_hint="uvicorn-runner-sys-path-src",
            )

        if (repo_root / "celeryconfig.py").is_file():
            status = await self._python_bootstrap_status(repo_root, Path("celeryconfig.py"))
            register_anchor(
                "celeryconfig.py",
                context="celery",
                anchor_type="python_src_bootstrap",
                status=status,
                repair_hint="celeryconfig-sys-path-src",
            )
            register_entrypoint(
                "celeryconfig.py",
                context="celery",
                config_anchor_path="celeryconfig.py",
                repair_hint="celeryconfig-sys-path-src",
            )

        if (repo_root / "gunicorn.conf.py").is_file():
            status = await self._gunicorn_anchor_status(repo_root, Path("gunicorn.conf.py"))
            register_anchor(
                "gunicorn.conf.py",
                context="gunicorn",
                anchor_type="pythonpath",
                status=status,
                repair_hint="gunicorn-pythonpath-src",
            )
            register_entrypoint(
                "gunicorn.conf.py",
                context="gunicorn",
                config_anchor_path="gunicorn.conf.py",
                repair_hint="gunicorn-pythonpath-src",
            )

        if (repo_root / "alembic.ini").is_file():
            status = await self._alembic_anchor_status(repo_root, Path("alembic.ini"))
            register_anchor(
                "alembic.ini",
                context="alembic",
                anchor_type="prepend_sys_path",
                status=status,
                repair_hint="alembic-prepend-sys-path-src",
            )
            register_entrypoint(
                "alembic.ini",
                context="alembic",
                config_anchor_path="alembic.ini",
                repair_hint="alembic-prepend-sys-path-src",
            )

        pytest_target = self._select_pytest_startup_target(repo_root)
        pytest_status = await self._pytest_anchor_status(repo_root, pytest_target)
        register_anchor(
            pytest_target.as_posix(),
            context="pytest",
            anchor_type="pythonpath",
            status=pytest_status,
            repair_hint="pytest-pythonpath-src",
        )
        register_entrypoint(
            pytest_target.as_posix(),
            context="pytest",
            config_anchor_path=pytest_target.as_posix(),
            repair_hint="pytest-pythonpath-src",
        )

        topology.entrypoints = entrypoints
        topology.config_anchors = anchors
        topology.repair_hints = sorted(
            {
                item.repair_hint
                for item in [*entrypoints, *anchors]
                if item.repair_hint
            }
        )
        return topology

    def match_runtime_to_startup_topology(
        self,
        output: dict[str, Any],
        *,
        topology: StartupTopology | None,
        root_cause_class: str,
    ) -> dict[str, str]:
        if topology is None:
            return {}
        command = str(output.get("command", ""))
        stdout = str(output.get("stdout", ""))
        stderr = str(output.get("stderr", ""))
        combined = "\n".join(item for item in (command, stdout, stderr) if item)
        entry_by_path = {item.path: item for item in topology.entrypoints}
        anchor_by_path = {item.path: item for item in topology.config_anchors}

        def payload_for_entry(entry: StartupEntrypoint) -> dict[str, str]:
            payload = {
                "matched_entrypoint": entry.path,
                "startup_context": entry.context,
            }
            if entry.config_anchor_path:
                payload["matched_config_anchor"] = entry.config_anchor_path
            if entry.repair_hint:
                payload["repair_hint"] = entry.repair_hint
            return payload

        def payload_for_anchor(anchor: StartupConfigAnchor) -> dict[str, str]:
            payload = {
                "matched_config_anchor": anchor.path,
                "startup_context": anchor.context,
            }
            if anchor.repair_hint:
                payload["repair_hint"] = anchor.repair_hint
            matching_entry = next(
                (
                    item
                    for item in topology.entrypoints
                    if item.config_anchor_path == anchor.path or item.path == anchor.path
                ),
                None,
            )
            if matching_entry is not None:
                payload["matched_entrypoint"] = matching_entry.path
            return payload

        if root_cause_class == "config":
            env_anchor = next(
                (
                    item
                    for item in topology.config_anchors
                    if item.anchor_type == "env_template"
                ),
                None,
            )
            return payload_for_anchor(env_anchor) if env_anchor is not None else {}

        if root_cause_class != "startup":
            return {}

        mentioned_names = [
            Path(match.group(1)).name
            for match in re.finditer(r'File\s+["\']([^"\']+)["\']', combined)
        ]
        for name in mentioned_names:
            matching_entry = next((item for item in topology.entrypoints if Path(item.path).name == name), None)
            if matching_entry is not None:
                return payload_for_entry(matching_entry)
            matching_anchor = next((item for item in topology.config_anchors if Path(item.path).name == name), None)
            if matching_anchor is not None:
                return payload_for_anchor(matching_anchor)

        python_script = self._command_python_script(command)
        if python_script is not None:
            exact_entry = next(
                (item for item in topology.entrypoints if item.path == python_script or Path(item.path).name == Path(python_script).name),
                None,
            )
            if exact_entry is not None:
                return payload_for_entry(exact_entry)

        startup_context = self._detect_startup_context_text(combined)
        if startup_context == "django_manage":
            manage_entry = entry_by_path.get("manage.py")
            if manage_entry is not None:
                return payload_for_entry(manage_entry)
        if startup_context == "asgi":
            asgi_entry = entry_by_path.get("asgi.py")
            if asgi_entry is not None:
                return payload_for_entry(asgi_entry)
        if startup_context == "wsgi":
            wsgi_entry = entry_by_path.get("wsgi.py")
            if wsgi_entry is not None:
                return payload_for_entry(wsgi_entry)
        if startup_context == "uvicorn":
            if "asgi:" in combined and "asgi.py" in entry_by_path:
                return payload_for_entry(entry_by_path["asgi.py"])
            uvicorn_entry = next((item for item in topology.entrypoints if item.context == "uvicorn"), None)
            if uvicorn_entry is not None:
                return payload_for_entry(uvicorn_entry)
        if startup_context == "gunicorn":
            if "wsgi:" in combined and "wsgi.py" in entry_by_path:
                return payload_for_entry(entry_by_path["wsgi.py"])
            gunicorn_anchor = anchor_by_path.get("gunicorn.conf.py")
            if gunicorn_anchor is not None:
                return payload_for_anchor(gunicorn_anchor)
        if startup_context == "celery":
            celery_entry = next((item for item in topology.entrypoints if item.context == "celery"), None)
            if celery_entry is not None:
                return payload_for_entry(celery_entry)
        if startup_context == "alembic":
            alembic_entry = next((item for item in topology.entrypoints if item.context == "alembic"), None)
            if alembic_entry is not None:
                return payload_for_entry(alembic_entry)
        if startup_context == "pytest":
            pytest_entry = next((item for item in topology.entrypoints if item.context == "pytest"), None)
            if pytest_entry is not None:
                return payload_for_entry(pytest_entry)
        return {}

    def _filter_tools(
        self,
        permissions: ToolPermissionSet,
        tools: dict[str, AgentTool],
    ) -> dict[str, AgentTool]:
        if not permissions.allowed_tools:
            return tools
        return {
            name: tool
            for name, tool in tools.items()
            if permissions.allows(name)
        }

    def _tool_read_file(self) -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            path = str(tool_input.get("path", ""))
            target = Path(session.working_repo_root) / path
            if not target.is_file():
                return ToolResult(
                    tool_name="read_file",
                    ok=False,
                    error=f"File does not exist: {path}",
                )
            try:
                content = await self.file_store.read_text(target)
            except UnicodeDecodeError as exc:
                return ToolResult(
                    tool_name="read_file",
                    ok=False,
                    error=str(exc),
                )
            return ToolResult(
                tool_name="read_file",
                ok=True,
                output={"path": path, "content": content},
                summary=f"Read file {path} ({len(content)} chars).",
            )

        return AgentTool(
            spec=ToolSpec(
                name="read_file",
                description="Read a text file from the working repository.",
                input_schema={"path": "relative path string"},
            ),
            handler=handler,
        )

    def _tool_write_file(self) -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            path = str(tool_input.get("path", ""))
            content = str(tool_input.get("content", ""))
            target = Path(session.working_repo_root) / path
            old_content = await self.file_store.read_text(target) if target.exists() else ""
            await self.file_store.write_text(target, content)
            patch = self.patch_service.build_file_patch(path, old_content, content)
            return ToolResult(
                tool_name="write_file",
                ok=True,
                output={
                    "path": path,
                    "old_content": old_content,
                    "new_content": content,
                    "diff": patch.diff,
                },
                summary=f"Wrote file {path}.",
            )

        return AgentTool(
            spec=ToolSpec(
                name="write_file",
                description="Write a text file in the working repository.",
                input_schema={"path": "relative path string", "content": "full file content"},
                requires_write=True,
            ),
            handler=handler,
        )

    def _tool_search_repo(self, context: "RunContext") -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return ToolResult(
                    tool_name="search_repo",
                    ok=False,
                    error="query is required",
                )
            paths = [str(item) for item in tool_input.get("paths", []) if str(item)]
            quoted_paths = [shlex.quote(path) for path in paths]
            command = f"rg -n --no-heading -e {shlex.quote(query)}"
            if quoted_paths:
                command = " ".join([command, *quoted_paths])
            result = await self.command_runner.run(
                command=command,
                cwd=Path(session.working_repo_root),
                timeout_seconds=60,
                env=self._command_env(context),
            )
            matches: list[dict[str, Any]] = []
            if result.returncode in {0, 1}:
                for line in result.stdout.splitlines()[:100]:
                    parsed = self._parse_rg_match(line)
                    if parsed is not None:
                        matches.append(parsed)
                return ToolResult(
                    tool_name="search_repo",
                    ok=True,
                    output={"query": query, "matches": matches},
                    summary=f"Search returned {len(matches)} matches for {query!r}.",
                )
            return ToolResult(
                tool_name="search_repo",
                ok=False,
                error=result.stderr or f"rg failed with exit code {result.returncode}",
            )

        return AgentTool(
            spec=ToolSpec(
                name="search_repo",
                description="Search the repository using ripgrep.",
                input_schema={"query": "regex or plain text", "paths": "optional list of relative paths"},
            ),
            handler=handler,
        )

    def _tool_git_diff(self, context: "RunContext") -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            paths = [str(item) for item in tool_input.get("paths", []) if str(item)]
            command = "git diff --"
            if paths:
                command = " ".join([command, *[shlex.quote(path) for path in paths]])
            result = await self.command_runner.run(
                command=command,
                cwd=Path(session.working_repo_root),
                timeout_seconds=60,
                env=self._command_env(context),
            )
            if result.returncode != 0:
                return ToolResult(
                    tool_name="git_diff",
                    ok=False,
                    error=result.stderr or "git diff failed",
                )
            return ToolResult(
                tool_name="git_diff",
                ok=True,
                output={"diff": result.stdout, "paths": paths},
                summary=f"Collected git diff for {len(paths) or 'repository'} scope.",
            )

        return AgentTool(
            spec=ToolSpec(
                name="git_diff",
                description="Read git diff for the repository or specific paths.",
                input_schema={"paths": "optional list of relative paths"},
            ),
            handler=handler,
        )

    def _tool_ast_summary(self) -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            path = str(tool_input.get("path", ""))
            target = Path(session.working_repo_root) / path
            if not target.is_file():
                return ToolResult(
                    tool_name="ast_summary",
                    ok=False,
                    error=f"File does not exist: {path}",
                )
            try:
                content = await self.file_store.read_text(target)
            except Exception as exc:
                return ToolResult(
                    tool_name="ast_summary",
                    ok=False,
                    error=str(exc),
                )
            try:
                summary = self._summarize_source_file(path, content)
            except Exception as exc:
                return ToolResult(
                    tool_name="ast_summary",
                    ok=False,
                    error=str(exc),
                )
            return ToolResult(
                tool_name="ast_summary",
                ok=True,
                output={"path": path, **summary},
                summary=(
                    f"Source summary for {path}: {len(summary.get('functions', []))} functions, "
                    f"{len(summary.get('classes', []))} classes, {len(summary.get('imports', []))} imports."
                ),
            )

        return AgentTool(
            spec=ToolSpec(
                name="ast_summary",
                description="Build a lightweight structural summary for a source file.",
                input_schema={"path": "relative path to a source file"},
            ),
            handler=handler,
        )

    def _tool_run_static_review(self, context: "RunContext") -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            paths = [str(item) for item in tool_input.get("paths", session.targets)]
            findings, artifacts = await self.static_tooling.review(
                repo_root=Path(session.working_repo_root),
                targets=paths,
                config=context.config,
                rules=context.rules,
                env=self._command_env(context),
                language_profile=context.language_profile,
            )
            return ToolResult(
                tool_name="run_static_review",
                ok=True,
                output={
                    "paths": paths,
                    "findings": [self._finding_to_dict(item) for item in findings],
                    "artifacts": artifacts,
                },
                summary=f"Static tooling produced {len(findings)} findings.",
            )

        return AgentTool(
            spec=ToolSpec(
                name="run_static_review",
                description="Run deterministic static analysis and language-aware tool augmentation on target files.",
                input_schema={"paths": "list of relative paths"},
            ),
            handler=handler,
        )

    def _summarize_source_file(self, path: str, content: str) -> dict[str, Any]:
        language = detect_snippet_language(path)
        if language == "python":
            tree = ast.parse(content or "\n")
            functions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            ]
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            return {
                "language": language,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "has_module_docstring": ast.get_docstring(tree, clean=False) is not None,
            }
        return {
            "language": language,
            "functions": self._match_symbols(
                content,
                {
                    "javascript": r"\b(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)|\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\(",
                    "typescript": r"\b(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)|\b(?:export\s+)?(?:const|function)\s+([A-Za-z_][A-Za-z0-9_]*)",
                    "go": r"\bfunc\s+(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)",
                    "rust": r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)",
                    "java": r"\b(?:public|private|protected)?\s*(?:static\s+)?[A-Za-z0-9_<>,\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                    "kotlin": r"\bfun\s+([A-Za-z_][A-Za-z0-9_]*)",
                }.get(language, r"$^"),
            ),
            "classes": self._match_symbols(
                content,
                {
                    "javascript": r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
                    "typescript": r"\b(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)",
                    "go": r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct\b",
                    "rust": r"\b(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)",
                    "java": r"\b(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)",
                    "kotlin": r"\b(?:class|interface|object|enum\s+class|sealed\s+class|data\s+class)\s+([A-Za-z_][A-Za-z0-9_]*)",
                }.get(language, r"$^"),
            ),
            "imports": self._match_symbols(
                content,
                {
                    "javascript": r"""(?:from\s+|require\()["']([^"']+)["']""",
                    "typescript": r"""(?:from\s+|require\()["']([^"']+)["']""",
                    "go": r'"([^"]+)"',
                    "rust": r"\b(?:use|mod)\s+([A-Za-z0-9_:]+)",
                    "java": r"\bimport\s+([A-Za-z0-9_.]+)",
                    "kotlin": r"\bimport\s+([A-Za-z0-9_.]+)",
                }.get(language, r"$^"),
            ),
            "has_module_docstring": False,
        }

    def _match_symbols(self, content: str, pattern: str) -> list[str]:
        if not pattern or pattern == r"$^":
            return []
        matches = re.findall(pattern, content, flags=re.MULTILINE)
        values: list[str] = []
        for match in matches:
            if isinstance(match, tuple):
                values.extend(item for item in match if item)
            elif match:
                values.append(match)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped[:50]

    def _tool_run_test_command(self, context: "RunContext") -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            command = str(tool_input.get("command", "")).strip()
            if not command:
                return ToolResult(
                    tool_name="run_test_command",
                    ok=False,
                    error="command is required",
                )
            try:
                result = await self.test_runner.run(
                    command=command,
                    repo_root=Path(session.working_repo_root),
                    timeout_seconds=context.config.dynamic_debug.timeout_seconds,
                    env=self._command_env(context),
                )
            except TypeError as exc:
                if "unexpected keyword argument 'env'" not in str(exc):
                    raise
                result = await self.test_runner.run(
                    command=command,
                    repo_root=Path(session.working_repo_root),
                    timeout_seconds=context.config.dynamic_debug.timeout_seconds,
                )
            return ToolResult(
                tool_name="run_test_command",
                ok=True,
                output={
                    "command": result.command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "duration_seconds": result.duration_seconds,
                    "timed_out": result.timed_out,
                },
                summary=f"Executed command {command!r} with exit code {result.returncode}.",
            )

        return AgentTool(
            spec=ToolSpec(
                name="run_test_command",
                description="Execute a test or repro command in the working repository.",
                input_schema={"command": "shell command string"},
            ),
            handler=handler,
        )

    def _tool_parse_traceback(self) -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            text = str(tool_input.get("text", ""))
            tracebacks = self.traceback_parser.parse(text)
            return ToolResult(
                tool_name="parse_traceback",
                ok=True,
                output={
                    "tracebacks": [
                        {
                            "exception_type": item.exception_type,
                            "message": item.message,
                            "text": item.text,
                        }
                        for item in tracebacks
                    ]
                },
                summary=f"Parsed {len(tracebacks)} traceback blocks.",
            )

        return AgentTool(
            spec=ToolSpec(
                name="parse_traceback",
                description="Parse Python traceback text into structured data.",
                input_schema={"text": "stdout/stderr combined text"},
            ),
            handler=handler,
        )

    def _tool_shell_readonly(self, context: "RunContext") -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            command = str(tool_input.get("command", "")).strip()
            if not self._is_safe_readonly_command(command):
                return ToolResult(
                    tool_name="shell_readonly",
                    ok=False,
                    error=f"Blocked unsafe shell command: {command}",
                )
            result = await self.command_runner.run(
                command=command,
                cwd=Path(session.working_repo_root),
                timeout_seconds=60,
                env=self._command_env(context),
            )
            return ToolResult(
                tool_name="shell_readonly",
                ok=result.returncode in {0, 1},
                output={
                    "command": result.command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timed_out": result.timed_out,
                },
                summary=f"Executed readonly shell command {command!r}.",
                error=None if result.returncode in {0, 1} else result.stderr,
            )

        return AgentTool(
            spec=ToolSpec(
                name="shell_readonly",
                description="Execute a readonly shell command under a safety allowlist.",
                input_schema={"command": "readonly shell command"},
            ),
            handler=handler,
        )

    def _tool_prepare_safe_patch(self, context: "RunContext") -> AgentTool:
        async def handler(session: AgentSession, tool_input: dict[str, Any]) -> ToolResult:
            payload_feedback = session.payload.get("feedback")
            if not isinstance(payload_feedback, FeedbackBundle):
                return ToolResult(
                    tool_name="prepare_safe_patch",
                    ok=False,
                    error="feedback payload is required to prepare a maintenance patch",
                )
            startup_topology = self._startup_topology_from_value(session.payload.get("startup_topology"))
            if startup_topology is None:
                startup_topology = await self.discover_startup_topology(Path(session.working_repo_root))
            paths = [str(item) for item in tool_input.get("paths", session.targets)]
            patch_data = await self._build_safe_patch_from_feedback(
                working_repo_root=Path(session.working_repo_root),
                feedback=payload_feedback,
                handoffs=[item for item in session.payload.get("handoffs", []) if isinstance(item, dict)],
                startup_topology=startup_topology,
                restrict_paths=set(paths) if paths else None,
            )
            return ToolResult(
                tool_name="prepare_safe_patch",
                ok=True,
                output=patch_data,
                summary=str(patch_data.get("summary", "Prepared safe patch candidate.")),
            )

        return AgentTool(
            spec=ToolSpec(
                name="prepare_safe_patch",
                description="Build a safe maintenance patch from static findings and handoffs.",
                input_schema={"paths": "optional list of relative paths to prioritize"},
                requires_write=True,
            ),
            handler=handler,
        )

    async def _build_safe_patch_from_feedback(
        self,
        *,
        working_repo_root: Path,
        feedback: FeedbackBundle,
        handoffs: list[dict[str, Any]],
        startup_topology: StartupTopology | None,
        restrict_paths: set[str] | None,
    ) -> dict[str, Any]:
        updated_contents: dict[str, str] = {}
        original_contents: dict[str, str] = {}
        unsupported_findings: list[str] = []
        unresolved_handoffs: list[dict[str, Any]] = []
        applied_rule_ids: set[str] = set()
        repair_scope: set[str] = set()
        auto_fixed_blockers: set[str] = set()

        grouped_findings: dict[str, list[Finding]] = defaultdict(list)
        for finding in feedback.static_findings:
            if finding.path and (restrict_paths is None or finding.path in restrict_paths):
                grouped_findings[finding.path].append(finding)
        for handoff in handoffs:
            path_candidates = [str(item) for item in handoff.get("affected_files", []) if str(item)]
            rule_id = str(handoff.get("metadata", {}).get("rule_id", "")) or None
            severity_raw = str(handoff.get("severity", Severity.LOW.value)).lower()
            severity = Severity(severity_raw) if severity_raw in {item.value for item in Severity} else Severity.LOW
            for path in path_candidates:
                if restrict_paths is not None and path not in restrict_paths:
                    continue
                if not rule_id:
                    continue
                grouped_findings[path].append(
                    Finding(
                        source_agent=AgentKind.MAINTENANCE,
                        severity=severity,
                        rule_id=rule_id,
                        message=str(handoff.get("description", handoff.get("title", rule_id))),
                        category="handoff",
                        path=path,
                    )
                )

        for relative_path, findings in grouped_findings.items():
            absolute_path = working_repo_root / relative_path
            if not absolute_path.exists():
                continue
            if relative_path not in original_contents:
                original_contents[relative_path] = await self.file_store.read_text(absolute_path)
            current_content = updated_contents.get(relative_path, original_contents[relative_path])
            transformed_content = current_content
            for finding in findings:
                updated = self._apply_autofix(relative_path, finding.rule_id, transformed_content)
                if updated is None:
                    unsupported_findings.append(f"{finding.rule_id} at {finding.path or 'unknown path'}")
                    continue
                transformed_content = updated
                applied_rule_ids.add(finding.rule_id)
            if transformed_content != current_content:
                updated_contents[relative_path] = transformed_content
                repair_scope.add("safe_autofix")

        dependency_repairs = self._extract_dependency_repairs(feedback, handoffs)
        if dependency_repairs:
            repairs_by_target: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            missing_target_repairs: list[dict[str, str]] = []
            for repair in dependency_repairs:
                scope = str(repair.get("scope", "runtime"))
                dependency_target = self._select_dependency_target(
                    working_repo_root,
                    scope=scope,
                )
                if dependency_target is None:
                    missing_target_repairs.append(repair)
                    continue
                repairs_by_target[(dependency_target.as_posix(), scope)].append(repair)

            for (dependency_relative, scope), repairs in repairs_by_target.items():
                dependency_path = Path(dependency_relative)
                if dependency_relative not in original_contents:
                    original_contents[dependency_relative] = (
                        await self.file_store.read_text(working_repo_root / dependency_path)
                        if (working_repo_root / dependency_path).exists()
                        else ""
                    )
                current_content = updated_contents.get(
                    dependency_relative,
                    original_contents[dependency_relative],
                )
                transformed_content = current_content
                handled_modules: set[str] = set()
                for repair in repairs:
                    updated = self._apply_dependency_repair(
                        dependency_relative,
                        transformed_content,
                        package_name=str(repair["package_name"]),
                        scope=scope,
                    )
                    if updated is None:
                        unresolved_handoffs.append(
                            {
                                "kind": "dependency",
                                "reason": "dependency-target-unsupported",
                                "module_name": repair["module_name"],
                                "package_name": repair["package_name"],
                                "message": repair["message"],
                                "scope": scope,
                            }
                        )
                        continue
                    transformed_content = updated
                    handled_modules.add(str(repair["module_name"]))
            if transformed_content != current_content:
                updated_contents[dependency_relative] = transformed_content
                repair_scope.add("dependency")
                auto_fixed_blockers.add("dependency")
                for repair in repairs:
                    if str(repair["module_name"]) not in handled_modules:
                        unresolved_handoffs.append(
                            {
                                "kind": "dependency",
                                "reason": "dependency-not-applied",
                                "module_name": repair["module_name"],
                                "package_name": repair["package_name"],
                                "message": repair["message"],
                                "scope": scope,
                            }
                        )

            for repair in missing_target_repairs:
                unresolved_handoffs.append(
                    {
                        "kind": "dependency",
                        "reason": "dependency-target-missing",
                        "module_name": repair["module_name"],
                        "package_name": repair["package_name"],
                        "message": repair["message"],
                        "scope": repair.get("scope", "runtime"),
                    }
                )

        env_repairs = self._extract_env_repairs(feedback, handoffs, startup_topology=startup_topology)
        if env_repairs:
            env_relative = self._select_env_target(working_repo_root)
            if env_relative not in original_contents:
                original_contents[env_relative] = (
                    await self.file_store.read_text(working_repo_root / env_relative)
                    if (working_repo_root / env_relative).exists()
                    else ""
                )
            current_content = updated_contents.get(env_relative, original_contents[env_relative])
            transformed_content = current_content
            handled_env_vars: set[str] = set()
            for repair in env_repairs:
                updated = self._apply_env_repair(
                    transformed_content,
                    env_var=str(repair["env_var"]),
                )
                if updated is None:
                    unresolved_handoffs.append(
                        {
                            "kind": "config",
                            "reason": "env-placeholder-unsupported",
                            "env_var": repair["env_var"],
                            "message": repair["message"],
                        }
                    )
                    continue
                transformed_content = updated
                handled_env_vars.add(str(repair["env_var"]))
            if transformed_content != current_content:
                updated_contents[env_relative] = transformed_content
                repair_scope.add("config")
                auto_fixed_blockers.add("config")
            for repair in env_repairs:
                if str(repair["env_var"]) not in handled_env_vars:
                    unresolved_handoffs.append(
                        {
                            "kind": "config",
                            "reason": "env-placeholder-not-applied",
                            "env_var": repair["env_var"],
                            "message": repair["message"],
                        }
                    )

        startup_repairs, startup_unresolved = self._extract_startup_repairs(
            feedback,
            handoffs,
            repo_root=working_repo_root,
            startup_topology=startup_topology,
        )
        unresolved_handoffs.extend(startup_unresolved)
        if startup_repairs:
            repairs_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
            unresolved_targets: list[dict[str, str]] = []
            for repair in startup_repairs:
                startup_target = self._select_startup_target(
                    working_repo_root,
                    repair_type=str(repair["repair_type"]),
                )
                if startup_target is None:
                    unresolved_targets.append(repair)
                    continue
                repairs_by_target[startup_target.as_posix()].append(repair)

            for startup_relative, repairs in repairs_by_target.items():
                if startup_relative not in original_contents:
                    original_contents[startup_relative] = (
                        await self.file_store.read_text(working_repo_root / startup_relative)
                        if (working_repo_root / startup_relative).exists()
                        else ""
                    )
                current_content = updated_contents.get(startup_relative, original_contents[startup_relative])
                transformed_content = current_content
                applied_repairs: set[str] = set()
                for repair in repairs:
                    updated = self._apply_startup_repair(
                        startup_relative,
                        transformed_content,
                        repair_type=str(repair["repair_type"]),
                    )
                    if updated is None:
                        unresolved_handoffs.append(
                            {
                                "kind": "startup",
                                "reason": "startup-repair-unsupported",
                                "repair_type": repair["repair_type"],
                                "message": repair["message"],
                            }
                        )
                        continue
                    transformed_content = updated
                    applied_repairs.add(str(repair["repair_type"]))
                if transformed_content != current_content:
                    updated_contents[startup_relative] = transformed_content
                    repair_scope.add("startup")
                    auto_fixed_blockers.add("startup")
                for repair in repairs:
                    if str(repair["repair_type"]) not in applied_repairs:
                        unresolved_handoffs.append(
                            {
                                "kind": "startup",
                                "reason": "startup-repair-not-applied",
                                "repair_type": repair["repair_type"],
                                "message": repair["message"],
                            }
                        )

            for repair in unresolved_targets:
                unresolved_handoffs.append(
                    {
                        "kind": "startup",
                        "reason": "startup-target-missing",
                        "repair_type": repair["repair_type"],
                        "message": repair["message"],
                    }
                )

        file_patches: list[dict[str, Any]] = []
        for relative_path, new_content in sorted(updated_contents.items()):
            old_content = original_contents[relative_path]
            patch = self.patch_service.build_file_patch(relative_path, old_content, new_content)
            file_patches.append(
                {
                    "path": patch.path,
                    "old_content": patch.old_content,
                    "new_content": patch.new_content,
                    "diff": patch.diff,
                }
            )

        suggestions = [
            f"Manual follow-up required for {item}" for item in unsupported_findings[:10]
        ]
        for unresolved in unresolved_handoffs[:10]:
            suggestions.append(
                "Manual follow-up required for "
                f"{unresolved.get('kind', 'unknown')} blocker: {unresolved.get('message', unresolved.get('reason', 'unknown'))}"
            )
        safe_fix_only = bool(applied_rule_ids) and all(
            self.safe_fix_policy.allows(rule_id) for rule_id in applied_rule_ids
        ) and not repair_scope.intersection({"dependency", "config", "startup", "runtime"})
        diff_text = "".join(item["diff"] for item in file_patches)
        repair_scope_list = sorted(repair_scope) if repair_scope else ["advisory"]
        return {
            "summary": (
                f"Generated {len(file_patches)} file patches from "
                f"{len(feedback.all_findings)} upstream findings."
            ),
            "rationale": "Safe patch candidate prepared from autonomous maintenance handoffs.",
            "file_patches": file_patches,
            "validation_targets": sorted(updated_contents),
            "suggestions": suggestions,
            "applied": False,
            "diff_text": diff_text,
            "touched_files": sorted(updated_contents),
            "applied_rule_ids": sorted(applied_rule_ids),
            "unsupported_findings": unsupported_findings,
            "unresolved_handoffs": unresolved_handoffs,
            "auto_fixed_blockers": sorted(auto_fixed_blockers),
            "safe_fix_only": safe_fix_only,
            "publishable": bool(file_patches) and safe_fix_only,
            "patch_kind": (
                "safe_autofix_patch"
                if file_patches and repair_scope_list == ["safe_autofix"]
                else "targeted_blocker_patch"
                if file_patches
                else "advisory_only_patch"
            ),
            "repair_scope": repair_scope_list,
        }

    def _apply_autofix(self, relative_path: str, rule_id: str, content: str) -> str | None:
        if rule_id in {"missing-module-docstring", "D100"} and relative_path.endswith(".py"):
            return self.patch_service.add_module_docstring(content, "Maintained by Close-Devs.")
        if rule_id in {"missing-final-newline", "W292"}:
            return self.patch_service.ensure_final_newline(content)
        if rule_id in {"trailing-whitespace", "W291", "W293"}:
            return self.patch_service.strip_trailing_whitespace(content)
        if rule_id in {"excessive-eof-blank-lines", "W391"}:
            return self.patch_service.normalize_eof_blank_lines(content)
        return None

    def _extract_dependency_repairs(
        self,
        feedback: FeedbackBundle,
        handoffs: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        repairs: list[dict[str, str]] = []
        seen: set[str] = set()

        def register(module_name: str, message: str, *, scope: str) -> None:
            normalized = module_name.strip()
            scope_key = f"{scope}:{normalized}"
            if not normalized or scope_key in seen:
                return
            seen.add(scope_key)
            repairs.append(
                {
                    "module_name": normalized,
                    "package_name": self._dependency_package_name(normalized),
                    "message": message,
                    "scope": scope,
                }
            )

        for finding in [*feedback.dynamic_findings, *feedback.static_findings]:
            if finding.root_cause_class != "dependency":
                continue
            texts = [finding.message, *self._flatten_mapping_strings(finding.evidence)]
            for module_name in self._extract_missing_modules(texts):
                register(
                    module_name,
                    finding.message,
                    scope=self._dependency_repair_scope(
                        module_name=module_name,
                        texts=texts,
                        affected_files=[finding.path] if finding.path else [],
                        root_cause_class=finding.root_cause_class,
                        category=finding.category,
                    ),
                )

        for handoff in handoffs:
            handoff_kind = str(handoff.get("kind", "")).lower()
            metadata = handoff.get("metadata", {})
            root_cause_class = str(metadata.get("root_cause_class", "")).lower() if isinstance(metadata, dict) else ""
            if handoff_kind != "dependency" and root_cause_class != "dependency":
                continue
            texts = [
                str(handoff.get("title", "")),
                str(handoff.get("description", "")),
                str(handoff.get("recommended_change", "")),
            ]
            for evidence_item in handoff.get("evidence", []):
                if isinstance(evidence_item, dict):
                    texts.extend(
                        [
                            str(evidence_item.get("summary", "")),
                            *self._flatten_mapping_strings(evidence_item.get("data")),
                        ]
                    )
            for module_name in self._extract_missing_modules(texts):
                register(
                    module_name,
                    str(handoff.get("description", handoff.get("title", module_name))),
                    scope=self._dependency_repair_scope(
                        module_name=module_name,
                        texts=texts,
                        affected_files=[str(item) for item in handoff.get("affected_files", []) if str(item)],
                        root_cause_class=root_cause_class or handoff_kind,
                        category=str(metadata.get("category", "")) if isinstance(metadata, dict) else "",
                    ),
                )

        return repairs

    def _extract_env_repairs(
        self,
        feedback: FeedbackBundle,
        handoffs: list[dict[str, Any]],
        startup_topology: StartupTopology | None = None,
    ) -> list[dict[str, str]]:
        repairs: list[dict[str, str]] = []
        seen: set[str] = set()
        if startup_topology is not None and not any(
            item.anchor_type == "env_template" for item in startup_topology.config_anchors
        ):
            return repairs

        def register(env_var: str, message: str) -> None:
            normalized = env_var.strip().upper()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            repairs.append({"env_var": normalized, "message": message})

        for finding in [*feedback.dynamic_findings, *feedback.static_findings]:
            if finding.root_cause_class != "config":
                continue
            for env_var in self._extract_missing_env_vars([finding.message, *self._flatten_mapping_strings(finding.evidence)]):
                register(env_var, finding.message)

        for handoff in handoffs:
            handoff_kind = str(handoff.get("kind", "")).lower()
            metadata = handoff.get("metadata", {})
            root_cause_class = str(metadata.get("root_cause_class", "")).lower() if isinstance(metadata, dict) else ""
            if handoff_kind != "config" and root_cause_class != "config":
                continue
            texts = [
                str(handoff.get("title", "")),
                str(handoff.get("description", "")),
                str(handoff.get("recommended_change", "")),
            ]
            for evidence_item in handoff.get("evidence", []):
                if isinstance(evidence_item, dict):
                    texts.extend(
                        [
                            str(evidence_item.get("summary", "")),
                            *self._flatten_mapping_strings(evidence_item.get("data")),
                        ]
                    )
            for env_var in self._extract_missing_env_vars(texts):
                register(env_var, str(handoff.get("description", handoff.get("title", env_var))))

        return repairs

    def _extract_startup_repairs(
        self,
        feedback: FeedbackBundle,
        handoffs: list[dict[str, Any]],
        *,
        repo_root: Path,
        startup_topology: StartupTopology | None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        repairs: list[dict[str, str]] = []
        unresolved: list[dict[str, str]] = []
        seen_repairs: set[str] = set()
        seen_unresolved: set[tuple[str, str]] = set()

        def register_repair(repair_type: str, message: str) -> None:
            if repair_type in seen_repairs:
                return
            seen_repairs.add(repair_type)
            repairs.append({"repair_type": repair_type, "message": message})

        def register_unresolved(reason: str, message: str) -> None:
            key = (reason, message)
            if key in seen_unresolved:
                return
            seen_unresolved.add(key)
            unresolved.append(
                {
                    "kind": "startup",
                    "reason": reason,
                    "message": message,
                    "guidance": self._startup_unresolved_guidance(reason),
                }
            )

        def evaluate_texts(
            texts: list[str],
            message: str,
            *,
            output_like: dict[str, Any] | None = None,
            topology_gated: bool = False,
        ) -> None:
            combined = "\n".join(item for item in texts if item)
            if not combined:
                return
            if startup_topology is not None and topology_gated:
                match = self.match_runtime_to_startup_topology(
                    output_like or {"stdout": combined, "stderr": combined},
                    topology=startup_topology,
                    root_cause_class="startup",
                )
                repair_hint = str(match.get("repair_hint", "")).strip()
                if repair_hint:
                    register_repair(repair_hint, message)
                    return
            if self._is_startup_argument_mismatch_text(combined):
                register_unresolved("startup-arguments-unsupported", message)
                return
            if self._needs_alembic_prepend_sys_path_repair(texts, repo_root):
                register_repair("alembic-prepend-sys-path-src", message)
                return
            if self._needs_gunicorn_pythonpath_repair(texts, repo_root):
                register_repair("gunicorn-pythonpath-src", message)
                return
            if self._needs_uvicorn_runner_src_bootstrap_repair(texts, repo_root):
                register_repair("uvicorn-runner-sys-path-src", message)
                return
            if self._needs_celeryconfig_src_bootstrap_repair(texts, repo_root):
                register_repair("celeryconfig-sys-path-src", message)
                return
            if self._needs_manage_py_src_bootstrap_repair(texts, repo_root):
                register_repair("managepy-sys-path-src", message)
                return
            if self._needs_asgi_entrypoint_src_bootstrap_repair(texts, repo_root):
                register_repair("asgi-entrypoint-sys-path-src", message)
                return
            if self._needs_wsgi_entrypoint_src_bootstrap_repair(texts, repo_root):
                register_repair("wsgi-entrypoint-sys-path-src", message)
                return
            if self._needs_uvicorn_app_dir_guidance(texts, repo_root):
                register_unresolved("uvicorn-app-dir-unsupported", message)
                return
            if self._needs_celery_pythonpath_guidance(texts, repo_root):
                register_unresolved("celery-pythonpath-unsupported", message)
                return
            if self._needs_pytest_pythonpath_repair(texts, repo_root):
                register_repair("pytest-pythonpath-src", message)
                return
            register_unresolved("startup-blocker-unsupported", message)

        for finding in feedback.dynamic_findings:
            if finding.root_cause_class != "startup":
                continue
            output_like = {
                "command": str(finding.evidence.get("command", "")),
                "stdout": str(finding.evidence.get("stdout_excerpt", "")),
                "stderr": str(finding.evidence.get("stderr_excerpt", "")),
            }
            evaluate_texts(
                [finding.message, *self._flatten_mapping_strings(finding.evidence)],
                finding.message,
                output_like=output_like,
                topology_gated=startup_topology is not None,
            )

        for handoff in handoffs:
            handoff_kind = str(handoff.get("kind", "")).lower()
            source_agent = str(handoff.get("source_agent", "")).lower()
            metadata = handoff.get("metadata", {})
            root_cause_class = str(metadata.get("root_cause_class", "")).lower() if isinstance(metadata, dict) else ""
            if handoff_kind != "startup" and root_cause_class != "startup":
                continue
            if startup_topology is not None and source_agent and source_agent != AgentKind.DYNAMIC_DEBUG.value:
                continue
            texts = [
                str(handoff.get("title", "")),
                str(handoff.get("description", "")),
                str(handoff.get("recommended_change", "")),
            ]
            output_like = {
                "command": "",
                "stdout": "",
                "stderr": "",
            }
            for evidence_item in handoff.get("evidence", []):
                if isinstance(evidence_item, dict):
                    texts.extend(
                        [
                            str(evidence_item.get("summary", "")),
                            *self._flatten_mapping_strings(evidence_item.get("data")),
                        ]
                    )
                    if not output_like["command"]:
                        output_like["command"] = str(evidence_item.get("data", {}).get("command", "")) if isinstance(evidence_item.get("data"), dict) else ""
                    if not output_like["stderr"]:
                        output_like["stderr"] = str(evidence_item.get("data", {}).get("stderr_excerpt", "")) if isinstance(evidence_item.get("data"), dict) else ""
            if isinstance(metadata, dict) and metadata.get("repair_hint"):
                register_repair(str(metadata["repair_hint"]), str(handoff.get("description", handoff.get("title", "startup blocker"))))
                continue
            evaluate_texts(
                texts,
                str(handoff.get("description", handoff.get("title", "startup blocker"))),
                output_like=output_like,
                topology_gated=startup_topology is not None,
            )

        if repairs:
            unresolved = [
                item
                for item in unresolved
                if not (
                    item.get("reason") == "startup-blocker-unsupported"
                    and str(item.get("message", "")).startswith("Command failed with exit code")
                )
            ]

        return repairs, unresolved

    def _startup_topology_from_value(self, value: Any) -> StartupTopology | None:
        if isinstance(value, StartupTopology):
            return value
        if not isinstance(value, dict):
            return None
        return StartupTopology(
            repo_root=str(value.get("repo_root", "")),
            src_layout=bool(value.get("src_layout", False)),
            entrypoints=[
                StartupEntrypoint(
                    path=str(item.get("path", "")),
                    context=str(item.get("context", "")),
                    config_anchor_path=str(item.get("config_anchor_path")) if item.get("config_anchor_path") else None,
                    repair_hint=str(item.get("repair_hint")) if item.get("repair_hint") else None,
                    notes=[str(note) for note in item.get("notes", [])],
                )
                for item in value.get("entrypoints", [])
                if isinstance(item, dict)
            ],
            config_anchors=[
                StartupConfigAnchor(
                    path=str(item.get("path", "")),
                    context=str(item.get("context", "")),
                    anchor_type=str(item.get("anchor_type", "")),
                    status=str(item.get("status", "")),
                    repair_hint=str(item.get("repair_hint")) if item.get("repair_hint") else None,
                )
                for item in value.get("config_anchors", [])
                if isinstance(item, dict)
            ],
            repair_hints=[str(item) for item in value.get("repair_hints", [])],
        )

    def _select_dependency_target(self, repo_root: Path, *, scope: str = "runtime") -> Path | None:
        runtime_candidates = [
            Path("src/requirements.txt"),
            Path("requirements.txt"),
            Path("requirements/base.txt"),
        ]
        test_candidates = [
            Path("src/requirements-test.txt"),
            Path("requirements-test.txt"),
            Path("requirements/test.txt"),
            Path("src/requirements-dev.txt"),
            Path("requirements-dev.txt"),
            Path("requirements/dev.txt"),
        ]
        candidates = test_candidates if scope == "test" else runtime_candidates
        for candidate in candidates:
            if (repo_root / candidate).is_file():
                return candidate

        pyproject_path = repo_root / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                with pyproject_path.open("rb") as handle:
                    pyproject = tomllib.load(handle)
            except (OSError, tomllib.TOMLDecodeError):
                pyproject = {}
            project_section = pyproject.get("project", {}) if isinstance(pyproject, dict) else {}
            optional_dependencies = (
                project_section.get("optional-dependencies", {})
                if isinstance(project_section, dict)
                else {}
            )
            dependency_groups = (
                pyproject.get("dependency-groups", {})
                if isinstance(pyproject, dict)
                else {}
            )
            poetry_section = (
                pyproject.get("tool", {}).get("poetry", {})
                if isinstance(pyproject, dict)
                else {}
            )
            poetry_dependencies = (
                poetry_section.get("dependencies", {})
                if isinstance(poetry_section, dict)
                else {}
            )
            poetry_groups = poetry_section.get("group", {}) if isinstance(poetry_section, dict) else {}
            poetry_dev_dependencies = (
                poetry_section.get("dev-dependencies", {})
                if isinstance(poetry_section, dict)
                else {}
            )
            if scope == "test":
                if isinstance(optional_dependencies, dict) and any(
                    isinstance(optional_dependencies.get(name), list) and optional_dependencies.get(name)
                    for name in ("test", "dev")
                ):
                    return Path("pyproject.toml")
                if isinstance(dependency_groups, dict) and any(
                    isinstance(dependency_groups.get(name), list) and dependency_groups.get(name)
                    for name in ("test", "dev")
                ):
                    return Path("pyproject.toml")
                if isinstance(poetry_groups, dict) and any(
                    isinstance(poetry_groups.get(name), dict)
                    and isinstance(poetry_groups.get(name, {}).get("dependencies", {}), dict)
                    and poetry_groups.get(name, {}).get("dependencies", {})
                    for name in ("test", "dev")
                ):
                    return Path("pyproject.toml")
                if isinstance(poetry_dev_dependencies, dict):
                    return Path("pyproject.toml")
                runtime_target = self._select_dependency_target(repo_root, scope="runtime")
                if runtime_target is not None:
                    return runtime_target
            if isinstance(project_section, dict) and project_section:
                return Path("pyproject.toml")
            if isinstance(poetry_dependencies, dict) and any(key != "python" for key in poetry_dependencies):
                return Path("pyproject.toml")

        fallback = Path("requirements-test.txt") if scope == "test" else Path("requirements.txt")
        return fallback

    def _select_env_target(self, repo_root: Path) -> str:
        for candidate in (".env.example", ".env.sample", ".env.template", ".env.dist"):
            if (repo_root / candidate).exists():
                return candidate
        return ".env.example"

    def _select_startup_target(self, repo_root: Path, *, repair_type: str) -> Path | None:
        if repair_type == "alembic-prepend-sys-path-src":
            return Path("alembic.ini") if (repo_root / "alembic.ini").is_file() else None
        if repair_type == "gunicorn-pythonpath-src":
            return Path("gunicorn.conf.py") if (repo_root / "gunicorn.conf.py").is_file() else None
        if repair_type == "uvicorn-runner-sys-path-src":
            return self._select_uvicorn_runner_target(repo_root)
        if repair_type == "celeryconfig-sys-path-src":
            return Path("celeryconfig.py") if (repo_root / "celeryconfig.py").is_file() else None
        if repair_type == "managepy-sys-path-src":
            return Path("manage.py") if (repo_root / "manage.py").is_file() else None
        if repair_type == "asgi-entrypoint-sys-path-src":
            return Path("asgi.py") if (repo_root / "asgi.py").is_file() else None
        if repair_type == "wsgi-entrypoint-sys-path-src":
            return Path("wsgi.py") if (repo_root / "wsgi.py").is_file() else None
        if repair_type != "pytest-pythonpath-src":
            return None
        return self._select_pytest_startup_target(repo_root)

    def _select_pytest_startup_target(self, repo_root: Path) -> Path:
        pyproject_path = repo_root / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                with pyproject_path.open("rb") as handle:
                    tomllib.load(handle)
            except (OSError, tomllib.TOMLDecodeError):
                return Path("pytest.ini")
            return Path("pyproject.toml")
        return Path("pytest.ini")

    def _apply_dependency_repair(
        self,
        relative_path: str,
        content: str,
        *,
        package_name: str,
        scope: str = "runtime",
    ) -> str | None:
        if relative_path.endswith(".toml"):
            return self._ensure_pyproject_dependency(content, package_name, scope=scope)
        return self._ensure_requirements_dependency(content, package_name)

    def _apply_env_repair(self, content: str, *, env_var: str) -> str | None:
        line = f"{env_var}=\n"
        normalized_lines = {item.split("=", 1)[0].strip() for item in content.splitlines() if "=" in item}
        if env_var in normalized_lines:
            return content
        updated = content
        if updated and not updated.endswith("\n"):
            updated += "\n"
        updated += line
        return updated

    def _apply_startup_repair(
        self,
        relative_path: str,
        content: str,
        *,
        repair_type: str,
    ) -> str | None:
        if repair_type == "pytest-pythonpath-src" and relative_path.endswith(".toml"):
            return self._ensure_pyproject_pytest_pythonpath(content)
        if repair_type == "pytest-pythonpath-src" and Path(relative_path).name == "pytest.ini":
            return self._ensure_pytest_ini_pythonpath(content)
        if repair_type == "alembic-prepend-sys-path-src" and Path(relative_path).name == "alembic.ini":
            return self._ensure_alembic_prepend_sys_path(content)
        if repair_type == "gunicorn-pythonpath-src" and Path(relative_path).name == "gunicorn.conf.py":
            return self._ensure_gunicorn_pythonpath(content)
        if repair_type in {
            "uvicorn-runner-sys-path-src",
            "celeryconfig-sys-path-src",
            "managepy-sys-path-src",
            "asgi-entrypoint-sys-path-src",
            "wsgi-entrypoint-sys-path-src",
        } and relative_path.endswith(".py"):
            return self._ensure_python_src_bootstrap(content, relative_path=relative_path)
        return None

    def _ensure_requirements_dependency(self, content: str, package_name: str) -> str:
        lines = [line.rstrip() for line in content.splitlines()]
        normalized = {line.strip().split("#", 1)[0].strip().lower() for line in lines if line.strip()}
        if package_name.lower() in normalized:
            return content
        updated_lines = list(content.splitlines())
        updated_lines.append(package_name)
        return "\n".join(updated_lines).rstrip("\n") + "\n"

    def _ensure_pyproject_dependency(
        self,
        content: str,
        package_name: str,
        *,
        scope: str = "runtime",
    ) -> str | None:
        if self._toml_dependency_present(content, package_name):
            return content
        if scope == "test":
            for section_header in ("[project.optional-dependencies]", "[dependency-groups]"):
                for key in ("test", "dev"):
                    updated = self._ensure_toml_array_dependency_section(
                        content,
                        section_header=section_header,
                        key=key,
                        package_name=package_name,
                    )
                    if updated is not None:
                        return updated
            for section_header in (
                "[tool.poetry.group.test.dependencies]",
                "[tool.poetry.group.dev.dependencies]",
                "[tool.poetry.dev-dependencies]",
            ):
                updated = self._ensure_poetry_dependency_table(
                    content,
                    section_header=section_header,
                    package_name=package_name,
                )
                if updated is not None:
                    return updated
        pattern = re.compile(
            r"(?ms)(^\[project\][^\[]*?^\s*dependencies\s*=\s*\[)(.*?)(^\s*\])"
        )
        match = pattern.search(content)
        if match is None:
            project_section = re.search(r"(?ms)(^\[project\]\s*\n)(.*?)(?=^\[|\Z)", content)
            if project_section is not None:
                header, body = project_section.group(1), project_section.group(2)
                dependency_block = 'dependencies = [\n    "' + package_name + '",\n]\n'
                updated_block = f"{header}{dependency_block}{body}"
                return content[: project_section.start()] + updated_block + content[project_section.end() :]

            poetry_section = re.search(
                r"(?ms)(^\[tool\.poetry\.dependencies\]\s*\n)(.*?)(?=^\[|\Z)",
                content,
            )
            if poetry_section is not None:
                header, body = poetry_section.group(1), poetry_section.group(2)
                body = body.rstrip()
                if body:
                    body += "\n"
                updated_block = f'{header}{body}{package_name} = "*"\n'
                return content[: poetry_section.start()] + updated_block + content[poetry_section.end() :]
            return None
        prefix, body, suffix = match.group(1), match.group(2), match.group(3)
        new_entry = f'    "{package_name}",\n'
        if body.strip():
            body = body.rstrip()
            if not body.endswith("\n"):
                body += "\n"
        updated_block = f"{prefix}{body}{new_entry}{suffix}"
        return content[: match.start()] + updated_block + content[match.end() :]

    def _ensure_toml_array_dependency_section(
        self,
        content: str,
        *,
        section_header: str,
        key: str,
        package_name: str,
    ) -> str | None:
        section_name = re.escape(section_header.strip("[]"))
        key_name = re.escape(key)
        pattern = re.compile(
            rf"(?ms)(^\[{section_name}\][^\[]*?^\s*{key_name}\s*=\s*\[)(.*?)(^\s*\])"
        )
        match = pattern.search(content)
        if match is not None:
            prefix, body, suffix = match.group(1), match.group(2), match.group(3)
            new_entry = f'    "{package_name}",\n'
            if body.strip():
                body = body.rstrip()
                if not body.endswith("\n"):
                    body += "\n"
            updated_block = f"{prefix}{body}{new_entry}{suffix}"
            return content[: match.start()] + updated_block + content[match.end() :]
        section_pattern = re.compile(rf"(?ms)(^\[{section_name}\]\s*\n)(.*?)(?=^\[|\Z)")
        section_match = section_pattern.search(content)
        if section_match is None:
            return None
        header, body = section_match.group(1), section_match.group(2)
        body = body.rstrip()
        if body:
            body += "\n"
        new_block = f'{header}{body}{key} = [\n    "{package_name}",\n]\n'
        return content[: section_match.start()] + new_block + content[section_match.end() :]

    def _ensure_poetry_dependency_table(
        self,
        content: str,
        *,
        section_header: str,
        package_name: str,
    ) -> str | None:
        section_name = re.escape(section_header.strip("[]"))
        section_pattern = re.compile(rf"(?ms)(^\[{section_name}\]\s*\n)(.*?)(?=^\[|\Z)")
        section_match = section_pattern.search(content)
        if section_match is None:
            return None
        header, body = section_match.group(1), section_match.group(2)
        if re.search(rf"(?m)^{re.escape(package_name)}\s*=", body):
            return content
        body = body.rstrip()
        if body:
            body += "\n"
        new_block = f'{header}{body}{package_name} = "*"\n'
        return content[: section_match.start()] + new_block + content[section_match.end() :]

    def _toml_dependency_present(self, content: str, package_name: str) -> bool:
        lowered = package_name.lower()
        return (
            f'"{lowered}"' in content.lower()
            or bool(re.search(rf"(?m)^{re.escape(package_name)}\s*=", content))
        )

    def _ensure_pyproject_pytest_pythonpath(self, content: str) -> str:
        section_pattern = re.compile(r"(?ms)(^\[tool\.pytest\.ini_options\]\s*\n)(.*?)(?=^\[|\Z)")
        section_match = section_pattern.search(content)
        if section_match is None:
            updated = content.rstrip()
            if updated:
                updated += "\n\n"
            updated += '[tool.pytest.ini_options]\npythonpath = ["src"]\n'
            return updated

        header, body = section_match.group(1), section_match.group(2)
        if re.search(r'(?m)^\s*pythonpath\s*=\s*\[[^\]]*["\']src["\']', body):
            return content

        pythonpath_match = re.search(r"(?m)^(\s*pythonpath\s*=\s*\[)([^\]]*)(\])", body)
        if pythonpath_match is not None:
            existing_items = pythonpath_match.group(2).strip()
            if existing_items:
                updated_items = f'{existing_items}, "src"'
            else:
                updated_items = '"src"'
            updated_body = (
                body[: pythonpath_match.start()]
                + pythonpath_match.group(1)
                + updated_items
                + pythonpath_match.group(3)
                + body[pythonpath_match.end() :]
            )
        else:
            updated_body = body.rstrip()
            if updated_body:
                updated_body += "\n"
            updated_body += 'pythonpath = ["src"]\n'
        updated_block = f"{header}{updated_body}"
        return content[: section_match.start()] + updated_block + content[section_match.end() :]

    def _ensure_pytest_ini_pythonpath(self, content: str) -> str:
        section_pattern = re.compile(r"(?ms)(^\[pytest\]\s*\n)(.*?)(?=^\[|\Z)")
        section_match = section_pattern.search(content)
        if section_match is None:
            updated = content.rstrip()
            if updated:
                updated += "\n\n"
            updated += "[pytest]\npythonpath = src\n"
            return updated

        header, body = section_match.group(1), section_match.group(2)
        if re.search(r"(?m)^\s*pythonpath\s*=\s*src(?:\s+.*)?$", body):
            return content
        updated_body = body.rstrip()
        if updated_body:
            updated_body += "\n"
        updated_body += "pythonpath = src\n"
        updated_block = f"{header}{updated_body}"
        return content[: section_match.start()] + updated_block + content[section_match.end() :]

    def _ensure_alembic_prepend_sys_path(self, content: str) -> str:
        section_pattern = re.compile(r"(?ms)(^\[alembic\]\s*\n)(.*?)(?=^\[|\Z)")
        section_match = section_pattern.search(content)
        if section_match is None:
            updated = content.rstrip()
            if updated:
                updated += "\n\n"
            updated += "[alembic]\nprepend_sys_path = src\n"
            return updated

        header, body = section_match.group(1), section_match.group(2)
        if re.search(r"(?m)^\s*prepend_sys_path\s*=\s*src(?:\s+.*)?$", body):
            return content
        updated_body = body.rstrip()
        if updated_body:
            updated_body += "\n"
        updated_body += "prepend_sys_path = src\n"
        updated_block = f"{header}{updated_body}"
        return content[: section_match.start()] + updated_block + content[section_match.end() :]

    def _ensure_gunicorn_pythonpath(self, content: str) -> str:
        if re.search(r'(?m)^\s*pythonpath\s*=\s*["\']src["\']\s*$', content):
            return content
        updated = content.rstrip()
        if updated:
            updated += "\n"
        updated += 'pythonpath = "src"\n'
        return updated

    def _ensure_python_src_bootstrap(self, content: str, *, relative_path: str) -> str:
        if "_CLOSE_DEVS_SRC_ROOT" in content:
            return content
        parent_depth = max(len(Path(relative_path).parts) - 1, 0)
        bootstrap = (
            "from pathlib import Path\n"
            "import sys\n\n"
            f"_CLOSE_DEVS_REPO_ROOT = Path(__file__).resolve().parents[{parent_depth}]\n"
            '_CLOSE_DEVS_SRC_ROOT = _CLOSE_DEVS_REPO_ROOT / "src"\n'
            "if _CLOSE_DEVS_SRC_ROOT.is_dir() and str(_CLOSE_DEVS_SRC_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(_CLOSE_DEVS_SRC_ROOT))\n\n"
        )
        lines = content.splitlines(keepends=True)
        insert_at = 0
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        if insert_at < len(lines) and re.match(r"#.*coding[:=]", lines[insert_at]):
            insert_at += 1
        return "".join([*lines[:insert_at], bootstrap, *lines[insert_at:]])

    async def _python_bootstrap_status(self, repo_root: Path, relative_path: Path) -> str:
        target = repo_root / relative_path
        if not target.is_file():
            return "missing"
        try:
            content = await self.file_store.read_text(target)
        except (OSError, UnicodeDecodeError):
            return "missing"
        return "present" if self._has_python_src_bootstrap(content) else "missing"

    async def _pytest_anchor_status(self, repo_root: Path, relative_path: Path) -> str:
        target = repo_root / relative_path
        if not target.exists():
            return "missing"
        try:
            content = await self.file_store.read_text(target)
        except (OSError, UnicodeDecodeError):
            return "missing"
        if target.suffix == ".toml":
            return "present" if self._pyproject_has_pytest_pythonpath(content) else "missing"
        if target.name == "pytest.ini":
            return "present" if self._pytest_ini_has_pythonpath(content) else "missing"
        return "missing"

    async def _alembic_anchor_status(self, repo_root: Path, relative_path: Path) -> str:
        target = repo_root / relative_path
        if not target.exists():
            return "missing"
        try:
            content = await self.file_store.read_text(target)
        except (OSError, UnicodeDecodeError):
            return "missing"
        return "present" if re.search(r"(?m)^\s*prepend_sys_path\s*=\s*src(?:\s+.*)?$", content) else "missing"

    async def _gunicorn_anchor_status(self, repo_root: Path, relative_path: Path) -> str:
        target = repo_root / relative_path
        if not target.exists():
            return "missing"
        try:
            content = await self.file_store.read_text(target)
        except (OSError, UnicodeDecodeError):
            return "missing"
        return "present" if re.search(r'(?m)^\s*pythonpath\s*=\s*["\']src["\']\s*$', content) else "missing"

    def _has_python_src_bootstrap(self, content: str) -> bool:
        return (
            "_CLOSE_DEVS_SRC_ROOT" in content
            or bool(re.search(r"sys\.path\.(?:insert|append)\([^)]*['\"]src['\"]", content))
            or bool(re.search(r"Path\(__file__\).*['\"]src['\"]", content))
        )

    def _pyproject_has_pytest_pythonpath(self, content: str) -> bool:
        return bool(re.search(r'(?m)^\s*pythonpath\s*=\s*\[[^\]]*["\']src["\']', content))

    def _pytest_ini_has_pythonpath(self, content: str) -> bool:
        return bool(re.search(r"(?m)^\s*pythonpath\s*=\s*src(?:\s+.*)?$", content))

    def _command_python_script(self, command: str) -> str | None:
        match = re.search(r"(?:^|\s)python(?:\d+(?:\.\d+)*)?\s+([^\s]+\.py)(?:\s|$)", command)
        if match:
            return Path(match.group(1)).name
        return None

    def _extract_missing_modules(self, texts: list[str]) -> list[str]:
        modules: list[str] = []
        patterns = (
            r"(?:No module named|ModuleNotFoundError:\s*No module named)\s+['\"]([^'\"]+)['\"]",
            r"could not import module\s+['\"]([^'\"]+)['\"]",
        )
        for text in texts:
            if not text:
                continue
            for pattern in patterns:
                modules.extend(
                    match.group(1)
                    for match in re.finditer(
                        pattern,
                        text,
                        re.IGNORECASE,
                    )
                )
        return modules

    def _needs_pytest_pythonpath_repair(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir():
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        combined = "\n".join(texts).lower()
        return (
            "importerror while importing test module" in combined
            or "conftest.py" in combined and "no module named" in combined
        )

    def _needs_alembic_prepend_sys_path_repair(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not (repo_root / "alembic.ini").is_file():
            return False
        if not self._mentions_alembic_context(texts):
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _needs_gunicorn_pythonpath_repair(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not (repo_root / "gunicorn.conf.py").is_file():
            return False
        if not self._mentions_gunicorn_context(texts):
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _needs_uvicorn_runner_src_bootstrap_repair(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not self._mentions_uvicorn_context(texts):
            return False
        if self._select_uvicorn_runner_target(repo_root) is None:
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _needs_celeryconfig_src_bootstrap_repair(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not self._mentions_celery_context(texts):
            return False
        if not (repo_root / "celeryconfig.py").is_file():
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _needs_manage_py_src_bootstrap_repair(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not (repo_root / "manage.py").is_file():
            return False
        if not self._mentions_manage_context(texts):
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _needs_asgi_entrypoint_src_bootstrap_repair(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not (repo_root / "asgi.py").is_file():
            return False
        if not self._mentions_asgi_entrypoint_context(texts):
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _needs_wsgi_entrypoint_src_bootstrap_repair(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not (repo_root / "wsgi.py").is_file():
            return False
        if not self._mentions_wsgi_entrypoint_context(texts):
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _needs_uvicorn_app_dir_guidance(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not self._mentions_uvicorn_context(texts):
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _needs_celery_pythonpath_guidance(self, texts: list[str], repo_root: Path) -> bool:
        if not repo_root.joinpath("src").is_dir() or not self._mentions_celery_context(texts):
            return False
        for module_name in self._extract_missing_modules(texts):
            if self._local_src_module_path(repo_root, module_name) is not None:
                return True
        return False

    def _is_startup_argument_mismatch_text(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in (
                "unrecognized arguments:",
                "the following arguments are required:",
                "error: argument",
            )
        )

    def _mentions_alembic_context(self, texts: list[str]) -> bool:
        combined = "\n".join(texts).lower()
        return "alembic" in combined or "env.py" in combined

    def _mentions_gunicorn_context(self, texts: list[str]) -> bool:
        return "gunicorn" in "\n".join(texts).lower()

    def _mentions_uvicorn_context(self, texts: list[str]) -> bool:
        combined = "\n".join(texts).lower()
        return "uvicorn" in combined or "asgi app" in combined

    def _mentions_celery_context(self, texts: list[str]) -> bool:
        return "celery" in "\n".join(texts).lower()

    def _mentions_manage_context(self, texts: list[str]) -> bool:
        combined = "\n".join(texts).lower()
        return "manage.py" in combined or "django.core.management" in combined

    def _mentions_asgi_entrypoint_context(self, texts: list[str]) -> bool:
        combined = "\n".join(texts).lower()
        return "asgi.py" in combined or " asgi:" in combined or self._mentions_uvicorn_context(texts)

    def _mentions_wsgi_entrypoint_context(self, texts: list[str]) -> bool:
        combined = "\n".join(texts).lower()
        return "wsgi.py" in combined or " wsgi:" in combined or self._mentions_gunicorn_context(texts)

    def _detect_startup_context_text(self, text: str) -> str:
        lowered = text.lower()
        if "manage.py" in lowered or "django.core.management" in lowered:
            return "django_manage"
        if "asgi.py" in lowered:
            return "asgi"
        if "wsgi.py" in lowered:
            return "wsgi"
        if "uvicorn" in lowered or "asgi app" in lowered:
            return "uvicorn"
        if "gunicorn" in lowered:
            return "gunicorn"
        if "alembic" in lowered or "env.py" in lowered:
            return "alembic"
        if "celery" in lowered:
            return "celery"
        if "pytest" in lowered or "importerror while importing test module" in lowered:
            return "pytest"
        return "generic"

    def _select_uvicorn_runner_target(self, repo_root: Path) -> Path | None:
        preferred_names = ("main.py", "app.py", "server.py", "run.py", "serve.py")
        candidates: list[Path] = []
        for candidate in repo_root.rglob("*.py"):
            if any(part.startswith(".") for part in candidate.relative_to(repo_root).parts):
                continue
            if any(part in {"reports", "state", ".venv", "__pycache__", "tests"} for part in candidate.relative_to(repo_root).parts):
                continue
            try:
                content = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "uvicorn.run(" not in content:
                continue
            relative = candidate.relative_to(repo_root)
            candidates.append(relative)
        if not candidates:
            return None
        for name in preferred_names:
            for candidate in candidates:
                if candidate.name == name:
                    return candidate
        return sorted(candidates, key=lambda item: (len(item.parts), item.as_posix()))[0]

    def _startup_unresolved_guidance(self, reason: str) -> str:
        guidance = {
            "startup-arguments-unsupported": (
                "Review the startup command contract and supply the required CLI arguments manually."
            ),
            "uvicorn-app-dir-unsupported": (
                "Adjust the uvicorn startup configuration to use the repository src layout, for example by setting "
                "an explicit app-dir or equivalent Python path bootstrap."
            ),
            "celery-pythonpath-unsupported": (
                "Adjust the celery worker startup configuration so the app module resolves from the src layout, for "
                "example by setting the working directory or Python path explicitly."
            ),
            "startup-blocker-unsupported": (
                "Review the Python startup/bootstrap entrypoint manually; Close-Devs could not apply a safe deterministic repair."
            ),
        }
        return guidance.get(reason, "Review the startup blocker manually.")

    def _local_src_module_path(self, repo_root: Path, module_name: str) -> Path | None:
        top_level = module_name.split(".", 1)[0]
        candidates = [
            repo_root / "src" / top_level / "__init__.py",
            repo_root / "src" / f"{top_level}.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _extract_missing_env_vars(self, texts: list[str]) -> list[str]:
        env_vars: list[str] = []
        patterns = [
            r"KeyError:\s*['\"]([A-Z][A-Z0-9_]+)['\"]",
            r"Missing environment variable[:\s]+([A-Z][A-Z0-9_]+)",
            r"environment variable\s+([A-Z][A-Z0-9_]+)\s+is required",
            r"(?m)^([A-Z][A-Z0-9_]+)\s*$\n\s+Field required\b",
        ]
        for text in texts:
            if not text:
                continue
            for pattern in patterns:
                env_vars.extend(
                    match.group(1)
                    for match in re.finditer(pattern, text, re.IGNORECASE)
                )
        return env_vars

    def _dependency_package_name(self, module_name: str) -> str:
        mapping = {
            "argon2": "argon2-cffi",
            "yaml": "PyYAML",
            "dotenv": "python-dotenv",
            "jwt": "PyJWT",
            "PIL": "Pillow",
            "bs4": "beautifulsoup4",
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "dateutil": "python-dateutil",
            "multipart": "python-multipart",
            "pydantic_settings": "pydantic-settings",
            "psycopg2": "psycopg2-binary",
            "pytest_mock": "pytest-mock",
            "pytest_asyncio": "pytest-asyncio",
            "pytest_cov": "pytest-cov",
            "faker": "Faker",
            "freezegun": "freezegun",
            "factory": "factory-boy",
            "factory_boy": "factory-boy",
            "responses": "responses",
            "respx": "respx",
            "hypothesis": "hypothesis",
        }
        return mapping.get(module_name, mapping.get(module_name.lower(), module_name))

    def _dependency_repair_scope(
        self,
        *,
        module_name: str | None,
        texts: list[str],
        affected_files: list[str],
        root_cause_class: str | None,
        category: str,
    ) -> str:
        normalized_module = (module_name or "").strip().lower()
        test_modules = (
            "pytest",
            "pytest_",
            "hypothesis",
            "faker",
            "freezegun",
            "factory",
            "factory_boy",
            "responses",
            "respx",
            "coverage",
        )
        if normalized_module.startswith(test_modules):
            return "test"
        if (root_cause_class or "").lower() == "test" or category.lower() == "test":
            return "test"
        if any(self._is_test_path(path) for path in affected_files if path):
            return "test"
        combined = "\n".join(texts).lower()
        if "fixture" in combined and "not found" in combined:
            return "test"
        return "runtime"

    def _is_test_path(self, path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        return (
            normalized.startswith("tests/")
            or normalized.endswith("/tests")
            or normalized.endswith("_test.py")
            or normalized.endswith("test.py")
        )

    def _flatten_mapping_strings(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            flattened: list[str] = []
            for item in value.values():
                flattened.extend(self._flatten_mapping_strings(item))
            return flattened
        if isinstance(value, list):
            flattened = []
            for item in value:
                flattened.extend(self._flatten_mapping_strings(item))
            return flattened
        return [str(value)]

    def _finding_to_dict(self, finding: Finding) -> dict[str, Any]:
        return {
            "source_agent": finding.source_agent.value,
            "severity": finding.severity.value,
            "rule_id": finding.rule_id,
            "message": finding.message,
            "category": finding.category,
            "path": finding.path,
            "line": finding.line,
            "symbol": finding.symbol,
            "evidence": finding.evidence,
            "fingerprint": finding.fingerprint,
            "state": finding.state,
        }

    def _command_env(self, context: "RunContext") -> dict[str, str] | None:
        if context.execution_environment is None:
            return None
        return context.execution_environment.command_env()

    def _parse_rg_match(self, line: str) -> dict[str, Any] | None:
        match = re.match(r"^(?P<path>[^:]+):(?P<line>\d+):(?P<text>.*)$", line)
        if match is None:
            return None
        return {
            "path": match.group("path"),
            "line": int(match.group("line")),
            "text": match.group("text"),
        }

    def _is_safe_readonly_command(self, command: str) -> bool:
        lowered = command.lower()
        blocked_fragments = [
            ">",
            ">>",
            " rm ",
            "mv ",
            " mv ",
            "cp ",
            " cp ",
            "chmod",
            "chown",
            "git checkout",
            "git switch",
            "git commit",
            "git push",
            "git merge",
            "git rebase",
            "sed -i",
        ]
        padded = f" {lowered} "
        return all(fragment not in padded for fragment in blocked_fragments)


from core.models import RunContext  # noqa: E402  # circular import guard
