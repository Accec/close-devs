from __future__ import annotations

import asyncio
import ast
from fnmatch import fnmatch
import json
from pathlib import Path
import re
import shlex
import shutil
from typing import Any

from core.config import AppConfig
from core.models import AgentKind, Finding, LanguageProfile, Severity, StaticToolAdapter
from tools.command_runner import CommandResult, CommandRunner
from tools.language_support import build_static_tool_adapters


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
        env: dict[str, str] | None = None,
        language_profile: LanguageProfile | None = None,
        tool_adapters: list[StaticToolAdapter] | None = None,
    ) -> tuple[list[Finding], dict[str, Any]]:
        findings = await asyncio.to_thread(
            self._internal_findings,
            repo_root,
            targets,
            config,
            rules,
        )
        artifacts: dict[str, Any] = {"external_tools": {}}
        adapters = list(tool_adapters or self._default_tool_adapters(repo_root, config, language_profile))
        tool_coroutines: list[asyncio.Future[tuple[str, CommandResult]] | asyncio.Task[tuple[str, CommandResult]]] = []
        for adapter in adapters:
            if not adapter.command:
                continue
            executable = shlex.split(adapter.command)[0]
            search_path = env.get("PATH") if env else None
            if shutil.which(executable, path=search_path) is None:
                artifacts["external_tools"][adapter.name] = {
                    "status": "missing",
                    "language": adapter.language,
                    "ecosystem": adapter.ecosystem,
                    "parser": adapter.parser,
                }
                continue
            if adapter.pass_targets:
                command_targets = targets or ["."]
                quoted_targets = [shlex.quote(target) for target in command_targets]
                full_command = " ".join([adapter.command, *quoted_targets])
            else:
                full_command = adapter.command
            tool_coroutines.append(
                asyncio.create_task(
                    self._run_external_tool(
                        tool_name=adapter.name,
                        command=full_command,
                        repo_root=repo_root,
                        timeout_seconds=config.dynamic_debug.timeout_seconds,
                        env=env,
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

    def _default_tool_adapters(
        self,
        repo_root: Path,
        config: AppConfig,
        language_profile: LanguageProfile | None,
    ) -> list[StaticToolAdapter]:
        if language_profile is not None:
            return build_static_tool_adapters(
                repo_root,
                language_profile=language_profile,
                python_commands={
                    "ruff": config.static_review.ruff_command,
                    "mypy": config.static_review.mypy_command,
                    "bandit": config.static_review.bandit_command,
                },
            )
        return [
            StaticToolAdapter(
                name="ruff",
                language="python",
                ecosystem="python",
                command=str(config.static_review.ruff_command or ""),
                parser="ruff",
            ),
            StaticToolAdapter(
                name="mypy",
                language="python",
                ecosystem="python",
                command=str(config.static_review.mypy_command or ""),
                parser="mypy",
            ),
            StaticToolAdapter(
                name="bandit",
                language="python",
                ecosystem="python",
                command=str(config.static_review.bandit_command or ""),
                parser="bandit",
            ),
        ]

    async def _run_external_tool(
        self,
        tool_name: str,
        command: str,
        repo_root: Path,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> tuple[str, CommandResult]:
        result = await self.runner.run(
            command,
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
            env=env,
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
                self._logic_smell_findings(
                    relative_path=relative_path,
                    tree=tree,
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

    def _logic_smell_findings(
        self,
        relative_path: str,
        tree: ast.AST,
    ) -> list[Finding]:
        findings: list[Finding] = []
        is_test_file = self._is_test_file(relative_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                findings.extend(self._exception_handler_findings(relative_path, node))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                findings.extend(self._mutable_default_findings(relative_path, node))
            if isinstance(node, ast.Assert) and not is_test_file:
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.MEDIUM,
                        rule_id="assert-used-for-runtime-validation",
                        message=(
                            "Runtime code uses assert for validation, which can be stripped "
                            "with optimization and should be replaced with explicit checks."
                        ),
                        category="correctness",
                        path=relative_path,
                        line=node.lineno,
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

    def _exception_handler_findings(
        self,
        relative_path: str,
        node: ast.Try,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for handler in node.handlers:
            if handler.type is None:
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.HIGH,
                        rule_id="bare-except",
                        message=(
                            "Bare except catches every failure mode and makes root-cause "
                            "analysis significantly harder."
                        ),
                        category="correctness",
                        path=relative_path,
                        line=handler.lineno,
                    )
                )
            elif isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"}:
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.MEDIUM,
                        rule_id="broad-exception-catch",
                        message=(
                            f"Exception handler catches `{handler.type.id}`, which can hide "
                            "unrelated failures and make behavior too broad."
                        ),
                        category="correctness",
                        path=relative_path,
                        line=handler.lineno,
                    )
                )
            if self._suppresses_exception(handler):
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.HIGH,
                        rule_id="swallowed-exception",
                        message=(
                            "Exception handler suppresses the original error with control-flow "
                            "only logic and no logging or re-raise."
                        ),
                        category="correctness",
                        path=relative_path,
                        line=handler.lineno,
                    )
                )
        return findings

    def _mutable_default_findings(
        self,
        relative_path: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[Finding]:
        findings: list[Finding] = []
        positional_defaults = [
            argument.arg
            for argument, default in zip(
                node.args.args[-len(node.args.defaults):],
                node.args.defaults,
                strict=False,
            )
            if self._is_mutable_default(default)
        ]
        keyword_defaults = [
            argument.arg
            for argument, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
                strict=False,
            )
            if default is not None and self._is_mutable_default(default)
        ]
        risky_arguments = positional_defaults + keyword_defaults
        if not risky_arguments:
            return findings
        findings.append(
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.HIGH,
                rule_id="mutable-default-argument",
                message=(
                    "Function uses mutable default argument(s): "
                    + ", ".join(risky_arguments)
                    + ". This can leak state across calls."
                ),
                category="correctness",
                path=relative_path,
                line=node.lineno,
                symbol=node.name,
                evidence={"arguments": risky_arguments},
            )
        )
        return findings

    def _suppresses_exception(self, handler: ast.ExceptHandler) -> bool:
        if not handler.body:
            return False
        meaningful_nodes = [
            node for node in handler.body if not isinstance(node, ast.Expr) or not self._is_literal_expr(node)
        ]
        if not meaningful_nodes:
            return True
        if any(isinstance(node, ast.Raise) for node in meaningful_nodes):
            return False
        if any(self._looks_like_logging(node) for node in meaningful_nodes):
            return False
        return all(
            isinstance(node, (ast.Pass, ast.Return, ast.Continue, ast.Break))
            for node in meaningful_nodes
        )

    def _looks_like_logging(self, node: ast.stmt) -> bool:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return False
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr in {
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "critical",
        }:
            return True
        return isinstance(func, ast.Name) and func.id in {"print", "warn"}

    def _is_literal_expr(self, node: ast.Expr) -> bool:
        return isinstance(node.value, ast.Constant)

    def _is_mutable_default(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.List, ast.Dict, ast.Set))

    def _is_test_file(self, relative_path: str) -> bool:
        normalized = relative_path.replace("\\", "/")
        return (
            normalized.startswith("tests/")
            or "/tests/" in normalized
            or normalized.endswith("_test.py")
            or normalized.endswith("test_.py")
            or normalized.endswith("conftest.py")
        )

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
        if tool_name == "bandit":
            return self._parse_bandit_output(result)
        if tool_name == "eslint":
            return self._parse_eslint_output(result)
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
        if result.returncode != 0:
            return [
                Finding(
                    source_agent=AgentKind.STATIC_REVIEW,
                    severity=Severity.LOW,
                    rule_id=f"{tool_name}-execution-failed",
                    message=f"{tool_name} exited with code {result.returncode} and returned non-structured output.",
                    category="tooling",
                    evidence={
                        "tool": tool_name,
                        "stdout": result.stdout[:2000],
                        "stderr": result.stderr[:2000],
                    },
                )
            ]
        return []

    def _parse_bandit_output(self, result: CommandResult) -> list[Finding]:
        text = result.stdout.strip() or result.stderr.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            message = "Bandit returned unparsable JSON output."
            if result.returncode != 0:
                message = f"{message} Exit code {result.returncode}."
            return [
                Finding(
                    source_agent=AgentKind.STATIC_REVIEW,
                    severity=Severity.MEDIUM,
                    rule_id="bandit-parse-error",
                    message=message,
                    category="tooling",
                    evidence={
                        "tool": "bandit",
                        "stdout": result.stdout[:2000],
                        "stderr": result.stderr[:2000],
                    },
                )
            ]

        results = payload.get("results", []) if isinstance(payload, dict) else []
        findings: list[Finding] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            severity = self._bandit_severity(item.get("issue_severity"))
            path = str(item.get("filename", "")).strip() or None
            if path and path.startswith("./"):
                path = path[2:]
            cwe = item.get("issue_cwe")
            evidence = {
                "tool": "bandit",
                "confidence": item.get("issue_confidence"),
                "test_name": item.get("test_name"),
                "more_info": item.get("more_info"),
                "code": str(item.get("code", "")).strip(),
            }
            if isinstance(cwe, dict):
                evidence["cwe"] = {
                    "id": cwe.get("id"),
                    "link": cwe.get("link"),
                }
            elif cwe is not None:
                evidence["cwe"] = cwe
            findings.append(
                Finding(
                    source_agent=AgentKind.STATIC_REVIEW,
                    severity=severity,
                    rule_id=str(item.get("test_id", "bandit-issue")),
                    message=str(item.get("issue_text", "Security issue reported by bandit.")),
                    category="security",
                    root_cause_class="application",
                    path=path,
                    line=int(item["line_number"]) if item.get("line_number") is not None else None,
                    evidence=evidence,
                )
            )

        if findings:
            return findings

        if result.returncode != 0:
            return [
                Finding(
                    source_agent=AgentKind.STATIC_REVIEW,
                    severity=Severity.MEDIUM,
                    rule_id="bandit-execution-failed",
                    message=f"Bandit exited with code {result.returncode} and did not return any issue records.",
                    category="tooling",
                    evidence={
                        "tool": "bandit",
                        "stdout": result.stdout[:2000],
                        "stderr": result.stderr[:2000],
                    },
                )
            ]
        return []

    def _bandit_severity(self, value: object) -> Severity:
        normalized = str(value or "").strip().lower()
        if normalized == "high":
            return Severity.HIGH
        if normalized == "medium":
            return Severity.MEDIUM
        return Severity.LOW

    def _parse_eslint_output(self, result: CommandResult) -> list[Finding]:
        text = result.stdout.strip() or result.stderr.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return [
                Finding(
                    source_agent=AgentKind.STATIC_REVIEW,
                    severity=Severity.LOW,
                    rule_id="eslint-parse-error",
                    message="ESLint returned unparsable JSON output.",
                    category="tooling",
                    evidence={"tool": "eslint"},
                )
            ]
        if not isinstance(payload, list):
            return []
        findings: list[Finding] = []
        for file_result in payload:
            if not isinstance(file_result, dict):
                continue
            path = str(file_result.get("filePath", "")).strip()
            if path.startswith("./"):
                path = path[2:]
            messages = file_result.get("messages", [])
            if not isinstance(messages, list):
                continue
            for message in messages:
                if not isinstance(message, dict):
                    continue
                severity = Severity.MEDIUM if int(message.get("severity", 1) or 1) >= 2 else Severity.LOW
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=severity,
                        rule_id=str(message.get("ruleId") or "eslint-issue"),
                        message=str(message.get("message") or "ESLint issue."),
                        category="lint",
                        path=path or None,
                        line=int(message["line"]) if message.get("line") is not None else None,
                        evidence={
                            "tool": "eslint",
                            "column": message.get("column"),
                            "endLine": message.get("endLine"),
                            "endColumn": message.get("endColumn"),
                            "language": "typescript" if path.endswith((".ts", ".tsx")) else "javascript",
                            "ecosystem": "node",
                        },
                    )
                )
        return findings

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
        if tool_name == "tsc" and (match := GENERIC_PATH_PATTERN.match(line)):
            return Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.HIGH,
                rule_id="tsc-error",
                message=match.group("msg").strip(),
                category="typing",
                path=match.group("path"),
                line=int(match.group("line")),
                root_cause_class="application",
                evidence={"tool": tool_name, "language": "typescript", "ecosystem": "node"},
            )
        if tool_name in {"go-test", "go-vet", "cargo-check", "cargo-clippy", "maven-compile", "gradle-check"} and (
            match := GENERIC_PATH_PATTERN.match(line)
        ):
            return Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.MEDIUM,
                rule_id=f"{tool_name}-issue",
                message=match.group("msg").strip(),
                category="correctness",
                path=match.group("path"),
                line=int(match.group("line")),
                root_cause_class="application",
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
