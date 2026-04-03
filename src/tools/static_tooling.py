from __future__ import annotations

import asyncio
import ast
from fnmatch import fnmatch
from pathlib import Path
import re
import shlex
import shutil
from typing import Any

from core.config import AppConfig
from core.models import AgentKind, Finding, Severity
from tools.command_runner import CommandResult, CommandRunner


RUFF_PATTERN = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+): (?P<rule>[A-Z]+\d+) (?P<msg>.+)$"
)
MYPY_PATTERN = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+): (?P<level>error|note): (?P<msg>.+?)(?:\s+\[(?P<rule>[^\]]+)\])?$"
)
GENERIC_PATH_PATTERN = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):(?P<msg>.+)$")


class StaticTooling:
    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or CommandRunner()

    async def review(
        self,
        repo_root: Path,
        targets: list[str],
        config: AppConfig,
        rules: dict[str, Any],
    ) -> tuple[list[Finding], dict[str, Any]]:
        findings = await asyncio.to_thread(
            self._internal_findings,
            repo_root,
            targets,
            config,
            rules,
        )
        artifacts: dict[str, Any] = {"external_tools": {}}

        external_commands = {
            "ruff": config.static_review.ruff_command,
            "mypy": config.static_review.mypy_command,
            "bandit": config.static_review.bandit_command,
        }
        tool_coroutines: list[asyncio.Future[tuple[str, CommandResult]] | asyncio.Task[tuple[str, CommandResult]]] = []
        for tool_name, command in external_commands.items():
            if not command:
                continue
            executable = shlex.split(command)[0]
            if shutil.which(executable) is None:
                artifacts["external_tools"][tool_name] = {"status": "missing"}
                continue
            command_targets = targets or ["."]
            quoted_targets = [shlex.quote(target) for target in command_targets]
            full_command = " ".join([command, *quoted_targets])
            tool_coroutines.append(
                asyncio.create_task(
                    self._run_external_tool(
                        tool_name=tool_name,
                        command=full_command,
                        repo_root=repo_root,
                        timeout_seconds=config.dynamic_debug.timeout_seconds,
                    )
                )
            )

        for tool_name, result in await asyncio.gather(*tool_coroutines):
            artifacts["external_tools"][tool_name] = {
                "status": "executed",
                "returncode": result.returncode,
            }
            findings.extend(self._parse_external_output(tool_name, result))

        return findings, artifacts

    async def _run_external_tool(
        self,
        tool_name: str,
        command: str,
        repo_root: Path,
        timeout_seconds: int,
    ) -> tuple[str, CommandResult]:
        result = await self.runner.run(
            command,
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        )
        return tool_name, result

    def _internal_findings(
        self,
        repo_root: Path,
        targets: list[str],
        config: AppConfig,
        rules: dict[str, Any],
    ) -> list[Finding]:
        findings: list[Finding] = []
        python_targets = [target for target in targets if target.endswith(".py")]
        for relative_path in python_targets:
            path = repo_root / relative_path
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            findings.extend(self._whitespace_findings(relative_path, content))
            if content and not content.endswith("\n"):
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.LOW,
                        rule_id="missing-final-newline",
                        message="File does not end with a newline.",
                        category="style",
                        path=relative_path,
                    )
                )

            try:
                tree = ast.parse(content or "\n")
            except SyntaxError as exc:
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.HIGH,
                        rule_id="syntax-error",
                        message=f"Syntax error: {exc.msg}",
                        category="correctness",
                        path=relative_path,
                        line=exc.lineno,
                    )
                )
                continue

            if ast.get_docstring(tree, clean=False) is None:
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.LOW,
                        rule_id="missing-module-docstring",
                        message="Python module is missing a top-level docstring.",
                        category="documentation",
                        path=relative_path,
                    )
                )

            findings.extend(
                self._complexity_findings(
                    relative_path=relative_path,
                    tree=tree,
                    max_complexity=config.static_review.max_complexity,
                )
            )
            findings.extend(
                self._architecture_findings(
                    relative_path=relative_path,
                    tree=tree,
                    rules=rules.get("architecture", {}),
                )
            )
        return findings

    def _whitespace_findings(self, relative_path: str, content: str) -> list[Finding]:
        findings: list[Finding] = []
        lines = content.splitlines()
        for line_number, line in enumerate(lines, start=1):
            if line.rstrip(" \t") != line:
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.LOW,
                        rule_id="trailing-whitespace",
                        message="Line contains trailing whitespace.",
                        category="style",
                        path=relative_path,
                        line=line_number,
                    )
                )
                break

        if content:
            trimmed = content.rstrip("\n")
            trailing_newlines = len(content) - len(trimmed)
            if trailing_newlines > 1:
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.LOW,
                        rule_id="excessive-eof-blank-lines",
                        message="File ends with more than one blank line.",
                        category="style",
                        path=relative_path,
                    )
                )
        return findings

    def _complexity_findings(
        self,
        relative_path: str,
        tree: ast.AST,
        max_complexity: int,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._estimate_complexity(node)
                if complexity > max_complexity:
                    findings.append(
                        Finding(
                            source_agent=AgentKind.STATIC_REVIEW,
                            severity=Severity.MEDIUM,
                            rule_id="high-branching-complexity",
                            message=(
                                f"Function complexity {complexity} exceeds configured "
                                f"limit {max_complexity}."
                            ),
                            category="complexity",
                            path=relative_path,
                            line=node.lineno,
                            symbol=node.name,
                            evidence={"complexity": complexity},
                        )
                    )
        return findings

    def _architecture_findings(
        self,
        relative_path: str,
        tree: ast.AST,
        rules: dict[str, Any],
    ) -> list[Finding]:
        findings: list[Finding] = []
        forbidden_rules = rules.get("forbidden_imports", [])
        if not forbidden_rules:
            return findings

        imports: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append((name.name, node.lineno))
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append((node.module, node.lineno))

        for import_rule in forbidden_rules:
            source_glob = str(import_rule.get("source_glob", ""))
            forbidden_prefix = str(import_rule.get("forbidden_prefix", ""))
            message = str(import_rule.get("message", "Forbidden import detected."))
            if source_glob and not fnmatch(relative_path, source_glob):
                continue
            for module_name, lineno in imports:
                if module_name.startswith(forbidden_prefix):
                    findings.append(
                        Finding(
                            source_agent=AgentKind.STATIC_REVIEW,
                            severity=Severity.HIGH,
                            rule_id="forbidden-import",
                            message=message,
                            category="architecture",
                            path=relative_path,
                            line=lineno,
                            symbol=module_name,
                        )
                    )
        return findings

    def _estimate_complexity(self, node: ast.AST) -> int:
        branch_nodes = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.BoolOp,
            ast.IfExp,
            ast.With,
            ast.AsyncWith,
            ast.Match,
            ast.comprehension,
        )
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, branch_nodes):
                complexity += 1
        return complexity

    def _parse_external_output(self, tool_name: str, result: CommandResult) -> list[Finding]:
        if result.returncode == 0 and not result.stdout.strip() and not result.stderr.strip():
            return []

        text = "\n".join(part for part in [result.stdout, result.stderr] if part)
        findings: list[Finding] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parsed = self._parse_line(tool_name, line)
            if parsed is None:
                continue
            findings.append(parsed)

        if findings:
            return findings

        return [
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.MEDIUM if result.returncode else Severity.LOW,
                rule_id=f"{tool_name}-output",
                message=line,
                category="tooling",
                evidence={"tool": tool_name},
            )
            for line in text.splitlines()
            if line.strip()
        ]

    def _parse_line(self, tool_name: str, line: str) -> Finding | None:
        if match := RUFF_PATTERN.match(line):
            return Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.MEDIUM,
                rule_id=match.group("rule"),
                message=match.group("msg"),
                category="lint",
                path=match.group("path"),
                line=int(match.group("line")),
                evidence={"tool": tool_name},
            )
        if match := MYPY_PATTERN.match(line):
            return Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.HIGH,
                rule_id=match.group("rule") or f"{tool_name}-error",
                message=match.group("msg"),
                category="typing",
                path=match.group("path"),
                line=int(match.group("line")),
                evidence={"tool": tool_name},
            )
        if match := GENERIC_PATH_PATTERN.match(line):
            return Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.MEDIUM,
                rule_id=f"{tool_name}-issue",
                message=match.group("msg").strip(),
                category="tooling",
                path=match.group("path"),
                line=int(match.group("line")),
                evidence={"tool": tool_name},
            )
        return None
