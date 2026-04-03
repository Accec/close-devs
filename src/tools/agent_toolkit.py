from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
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
    ToolPermissionSet,
    ToolResult,
    ToolSpec,
)
from tools.command_runner import CommandRunner
from tools.file_store import FileStore
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
                tree = ast.parse(content or "\n")
            except Exception as exc:
                return ToolResult(
                    tool_name="ast_summary",
                    ok=False,
                    error=str(exc),
                )
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
            return ToolResult(
                tool_name="ast_summary",
                ok=True,
                output={
                    "path": path,
                    "functions": functions,
                    "classes": classes,
                    "imports": imports,
                    "has_module_docstring": ast.get_docstring(tree, clean=False) is not None,
                },
                summary=(
                    f"AST summary for {path}: {len(functions)} functions, "
                    f"{len(classes)} classes, {len(imports)} imports."
                ),
            )

        return AgentTool(
            spec=ToolSpec(
                name="ast_summary",
                description="Build a lightweight AST summary for a Python file.",
                input_schema={"path": "relative path to a Python file"},
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
                description="Run deterministic static analysis on target files.",
                input_schema={"paths": "list of relative paths"},
            ),
            handler=handler,
        )

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
            paths = [str(item) for item in tool_input.get("paths", session.targets)]
            patch_data = await self._build_safe_patch_from_feedback(
                working_repo_root=Path(session.working_repo_root),
                feedback=payload_feedback,
                handoffs=[item for item in session.payload.get("handoffs", []) if isinstance(item, dict)],
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
        restrict_paths: set[str] | None,
    ) -> dict[str, Any]:
        updated_contents: dict[str, str] = {}
        original_contents: dict[str, str] = {}
        unsupported_findings: list[str] = []
        applied_rule_ids: set[str] = set()

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
        safe_fix_only = bool(applied_rule_ids) and all(
            self.safe_fix_policy.allows(rule_id) for rule_id in applied_rule_ids
        )
        diff_text = "".join(item["diff"] for item in file_patches)
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
            "safe_fix_only": safe_fix_only,
            "publishable": bool(file_patches) and safe_fix_only,
            "patch_kind": "safe_autofix_patch" if file_patches else "advisory_only_patch",
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
