from __future__ import annotations

import re
import tomllib
from pathlib import Path

from core.models import (
    DependencyAuditAdapter,
    LanguageAdapter,
    LanguageProfile,
    ProjectConfigAnchor,
    ProjectEntrypoint,
    ProjectTopology,
    StartupTopology,
    StaticToolAdapter,
)


SUPPORTED_LANGUAGE_ADAPTERS: tuple[LanguageAdapter, ...] = (
    LanguageAdapter(
        language="python",
        ecosystems=["python"],
        entrypoint_markers=["manage.py", "asgi.py", "wsgi.py", "gunicorn.conf.py", "alembic.ini", "pytest.ini"],
        config_markers=["pyproject.toml", "requirements.txt"],
        manifest_markers=["pyproject.toml", "requirements.txt", "src/requirements.txt"],
    ),
    LanguageAdapter(
        language="javascript",
        ecosystems=["node"],
        entrypoint_markers=["package.json"],
        config_markers=["tsconfig.json", "vite.config.ts", "vite.config.js", "next.config.js", "next.config.ts"],
        manifest_markers=["package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"],
    ),
    LanguageAdapter(
        language="typescript",
        ecosystems=["node"],
        entrypoint_markers=["package.json"],
        config_markers=["tsconfig.json", "vite.config.ts", "next.config.ts", "vitest.config.ts", "jest.config.ts"],
        manifest_markers=["package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"],
    ),
    LanguageAdapter(
        language="go",
        ecosystems=["go"],
        entrypoint_markers=["go.mod", "main.go"],
        config_markers=["go.mod"],
        manifest_markers=["go.mod", "go.sum"],
    ),
    LanguageAdapter(
        language="rust",
        ecosystems=["rust"],
        entrypoint_markers=["Cargo.toml", "src/main.rs"],
        config_markers=["Cargo.toml"],
        manifest_markers=["Cargo.toml", "Cargo.lock"],
    ),
    LanguageAdapter(
        language="java",
        ecosystems=["java"],
        entrypoint_markers=["pom.xml", "build.gradle", "build.gradle.kts"],
        config_markers=["pom.xml", "settings.gradle", "settings.gradle.kts"],
        manifest_markers=["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"],
    ),
    LanguageAdapter(
        language="kotlin",
        ecosystems=["java"],
        entrypoint_markers=["pom.xml", "build.gradle.kts"],
        config_markers=["pom.xml", "settings.gradle.kts"],
        manifest_markers=["pom.xml", "build.gradle.kts", "settings.gradle.kts"],
    ),
)

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
}

CODE_EXTENSIONS = frozenset(LANGUAGE_BY_SUFFIX)
SKIP_PARTS = {
    ".git",
    ".venv",
    "state",
    "reports",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    "target",
    ".gradle",
}


def supported_language_names() -> list[str]:
    return sorted({adapter.language for adapter in SUPPORTED_LANGUAGE_ADAPTERS})


def detect_language_profile(
    repo_root: Path,
    *,
    enabled_languages: list[str],
) -> LanguageProfile:
    normalized_enabled = {item.strip().lower() for item in enabled_languages if item.strip()}
    file_counts: dict[str, int] = {}
    ecosystems: set[str] = set()
    languages: set[str] = set()
    for path in repo_root.rglob("*"):
        if not path.is_file() or _should_skip(path, repo_root):
            continue
        language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
        if not language:
            continue
        file_counts[language] = file_counts.get(language, 0) + 1
        languages.add(language)
        ecosystems.add(_ecosystem_for_language(language))

    manifest_hits = _manifest_hits(repo_root)
    for ecosystem, present in manifest_hits.items():
        if present:
            ecosystems.add(ecosystem)
            languages.update(_languages_for_ecosystem(ecosystem))

    ordered_languages = sorted(
        languages,
        key=lambda item: (
            0 if item in normalized_enabled else 1,
            -file_counts.get(item, 0),
            item,
        ),
    )
    if not ordered_languages:
        ordered_languages = ["generic"]
    primary_language = ordered_languages[0]
    ordered_ecosystems = sorted(
        ecosystems,
        key=lambda item: (
            0 if any(language in normalized_enabled for language in _languages_for_ecosystem(item)) else 1,
            -sum(file_counts.get(language, 0) for language in _languages_for_ecosystem(item)),
            item,
        ),
    )
    if not ordered_ecosystems:
        ordered_ecosystems = ["generic"]
    primary_ecosystem = ordered_ecosystems[0]
    generic_review = not any(language in normalized_enabled for language in ordered_languages if language != "generic")
    enabled_adapters = [
        adapter.language
        for adapter in SUPPORTED_LANGUAGE_ADAPTERS
        if adapter.language in normalized_enabled and adapter.language in ordered_languages
    ]
    return LanguageProfile(
        primary_language=primary_language,
        languages=ordered_languages,
        primary_ecosystem=primary_ecosystem,
        ecosystems=ordered_ecosystems,
        language_file_counts=file_counts,
        enabled_adapters=sorted(set(enabled_adapters)),
        generic_review=generic_review,
    )


def build_project_topology(
    repo_root: Path,
    *,
    language_profile: LanguageProfile,
    startup_topology: StartupTopology | None = None,
) -> ProjectTopology:
    entrypoints: list[ProjectEntrypoint] = []
    anchors: list[ProjectConfigAnchor] = []
    dependency_manifests: set[str] = set()
    lockfiles: set[str] = set()
    package_roots: set[str] = set()
    test_roots: set[str] = set()

    for path in repo_root.rglob("*"):
        if not path.is_file() or _should_skip(path, repo_root):
            continue
        relative = path.relative_to(repo_root).as_posix()
        if path.suffix.lower() in CODE_EXTENSIONS:
            package_roots.add(str(path.parent.relative_to(repo_root).as_posix() or "."))
            if _is_test_path(path):
                test_roots.add(str(path.parent.relative_to(repo_root).as_posix() or "."))

    if startup_topology is not None:
        for item in startup_topology.entrypoints:
            entrypoints.append(
                ProjectEntrypoint(
                    path=item.path,
                    context=item.context,
                    language="python",
                    ecosystem="python",
                    config_anchor_path=item.config_anchor_path,
                    repair_hint=item.repair_hint,
                    notes=list(item.notes),
                )
            )
        for item in startup_topology.config_anchors:
            anchors.append(
                ProjectConfigAnchor(
                    path=item.path,
                    context=item.context,
                    language="python",
                    ecosystem="python",
                    anchor_type=item.anchor_type,
                    status=item.status,
                    repair_hint=item.repair_hint,
                )
            )

    _maybe_add_node_topology(repo_root, entrypoints, anchors, dependency_manifests, lockfiles)
    _maybe_add_go_topology(repo_root, entrypoints, anchors, dependency_manifests, lockfiles)
    _maybe_add_rust_topology(repo_root, entrypoints, anchors, dependency_manifests, lockfiles)
    _maybe_add_java_kotlin_topology(repo_root, entrypoints, anchors, dependency_manifests, lockfiles)
    _maybe_add_python_manifests(repo_root, dependency_manifests, lockfiles)

    repair_hints = sorted(
        {
            item.repair_hint
            for item in [*entrypoints, *anchors]
            if item.repair_hint
        }
    )
    return ProjectTopology(
        repo_root=str(repo_root),
        languages=list(language_profile.languages),
        ecosystems=list(language_profile.ecosystems),
        entrypoints=_dedupe_entrypoints(entrypoints),
        config_anchors=_dedupe_anchors(anchors),
        dependency_manifests=sorted(dependency_manifests),
        lockfiles=sorted(lockfiles),
        package_roots=sorted(item for item in package_roots if item)[:50],
        test_roots=sorted(item for item in test_roots if item)[:50],
        repair_hints=repair_hints,
    )


def build_static_tool_adapters(
    repo_root: Path,
    *,
    language_profile: LanguageProfile,
    python_commands: dict[str, str | None],
) -> list[StaticToolAdapter]:
    adapters: list[StaticToolAdapter] = []
    if "python" in language_profile.languages:
        if python_commands.get("ruff"):
            adapters.append(
                StaticToolAdapter(
                    name="ruff",
                    language="python",
                    ecosystem="python",
                    command=str(python_commands["ruff"]),
                    parser="ruff",
                )
            )
        if python_commands.get("mypy"):
            adapters.append(
                StaticToolAdapter(
                    name="mypy",
                    language="python",
                    ecosystem="python",
                    command=str(python_commands["mypy"]),
                    parser="mypy",
                )
            )
        if python_commands.get("bandit"):
            adapters.append(
                StaticToolAdapter(
                    name="bandit",
                    language="python",
                    ecosystem="python",
                    command=str(python_commands["bandit"]),
                    parser="bandit",
                )
            )
    if "node" in language_profile.ecosystems:
        adapters.extend(
            [
                StaticToolAdapter(
                    name="eslint",
                    language="typescript" if "typescript" in language_profile.languages else "javascript",
                    ecosystem="node",
                    command="eslint --format json .",
                    parser="eslint",
                    pass_targets=False,
                ),
                StaticToolAdapter(
                    name="tsc",
                    language="typescript",
                    ecosystem="node",
                    command="tsc --noEmit --pretty false",
                    parser="tsc",
                    pass_targets=False,
                ),
            ]
        )
    if "go" in language_profile.ecosystems:
        adapters.extend(
            [
                StaticToolAdapter(
                    name="go-test",
                    language="go",
                    ecosystem="go",
                    command="go test ./...",
                    parser="go",
                    pass_targets=False,
                ),
                StaticToolAdapter(
                    name="go-vet",
                    language="go",
                    ecosystem="go",
                    command="go vet ./...",
                    parser="go",
                    pass_targets=False,
                ),
            ]
        )
    if "rust" in language_profile.ecosystems:
        adapters.extend(
            [
                StaticToolAdapter(
                    name="cargo-check",
                    language="rust",
                    ecosystem="rust",
                    command="cargo check --message-format short",
                    parser="cargo",
                    pass_targets=False,
                ),
                StaticToolAdapter(
                    name="cargo-clippy",
                    language="rust",
                    ecosystem="rust",
                    command="cargo clippy --message-format short -- -D warnings",
                    parser="cargo",
                    pass_targets=False,
                ),
            ]
        )
    if "java" in language_profile.ecosystems:
        if (repo_root / "pom.xml").is_file():
            adapters.append(
                StaticToolAdapter(
                    name="maven-compile",
                    language="java",
                    ecosystem="java",
                    command="mvn -q -DskipTests compile",
                    parser="maven",
                    pass_targets=False,
                )
            )
        gradle_file = next(
            (
                candidate
                for candidate in ("gradlew",)
                if (repo_root / candidate).is_file()
            ),
            None,
        )
        if gradle_file is not None:
            adapters.append(
                StaticToolAdapter(
                    name="gradle-check",
                    language="kotlin" if "kotlin" in language_profile.languages else "java",
                    ecosystem="java",
                    command=f"./{gradle_file} check --dry-run",
                    parser="gradle",
                    pass_targets=False,
                )
            )
    return adapters


def build_dependency_audit_adapters(
    repo_root: Path,
    *,
    language_profile: LanguageProfile,
    mode: str,
    python_command: str | None,
) -> list[DependencyAuditAdapter]:
    if mode == "disabled":
        return []
    adapters: list[DependencyAuditAdapter] = []
    if "python" in language_profile.ecosystems and python_command:
        adapters.append(DependencyAuditAdapter(ecosystem="python", command=python_command, parser="pip-audit"))
    if "node" in language_profile.ecosystems and (repo_root / "package.json").is_file():
        if (repo_root / "pnpm-lock.yaml").is_file():
            adapters.append(DependencyAuditAdapter(ecosystem="node", command="pnpm audit --json", parser="npm"))
        elif (repo_root / "yarn.lock").is_file():
            adapters.append(DependencyAuditAdapter(ecosystem="node", command="yarn npm audit --json", parser="npm"))
        else:
            adapters.append(DependencyAuditAdapter(ecosystem="node", command="npm audit --json", parser="npm"))
    if "rust" in language_profile.ecosystems and (repo_root / "Cargo.toml").is_file():
        adapters.append(DependencyAuditAdapter(ecosystem="rust", command="cargo audit --json", parser="cargo-audit"))
    if "go" in language_profile.ecosystems and (repo_root / "go.mod").is_file():
        adapters.append(DependencyAuditAdapter(ecosystem="go", command="govulncheck -json ./...", parser="govulncheck"))
    if "java" in language_profile.ecosystems and (
        (repo_root / "pom.xml").is_file()
        or (repo_root / "build.gradle").is_file()
        or (repo_root / "build.gradle.kts").is_file()
    ):
        adapters.append(
            DependencyAuditAdapter(
                ecosystem="java",
                command="dependency-check.sh --format JSON --scan . --out .close-devs-dependency-check",
                parser="dependency-check",
            )
        )
    return adapters


def detect_snippet_language(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    if suffix == ".go":
        return "go"
    if suffix == ".rs":
        return "rust"
    if suffix == ".java":
        return "java"
    if suffix in {".kt", ".kts"}:
        return "kotlin"
    if suffix == ".py":
        return "python"
    if suffix == ".toml":
        return "toml"
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".ini":
        return "ini"
    if suffix == ".xml":
        return "xml"
    if suffix == ".md":
        return "markdown"
    return "text"


def _maybe_add_node_topology(
    repo_root: Path,
    entrypoints: list[ProjectEntrypoint],
    anchors: list[ProjectConfigAnchor],
    dependency_manifests: set[str],
    lockfiles: set[str],
) -> None:
    package_json = repo_root / "package.json"
    if not package_json.is_file():
        return
    dependency_manifests.add("package.json")
    payload = _read_json_like(package_json)
    scripts = payload.get("scripts", {}) if isinstance(payload, dict) else {}
    if isinstance(scripts, dict):
        for name in sorted(scripts):
            entrypoints.append(
                ProjectEntrypoint(
                    path="package.json",
                    context=f"node_script:{name}",
                    language="typescript" if (repo_root / "tsconfig.json").is_file() else "javascript",
                    ecosystem="node",
                    config_anchor_path="package.json",
                    notes=[str(scripts[name])],
                )
            )
    for candidate in (
        "tsconfig.json",
        "vite.config.ts",
        "vite.config.js",
        "next.config.ts",
        "next.config.js",
        "eslint.config.js",
        ".eslintrc.json",
        "jest.config.ts",
        "jest.config.js",
        "vitest.config.ts",
        "vitest.config.js",
    ):
        path = repo_root / candidate
        if path.is_file():
            anchors.append(
                ProjectConfigAnchor(
                    path=candidate,
                    context="node",
                    language="typescript" if candidate.endswith((".ts", ".tsx", ".json")) else "javascript",
                    ecosystem="node",
                    anchor_type="node-config",
                    status="present",
                )
            )
    for lockfile in ("package-lock.json", "pnpm-lock.yaml", "yarn.lock"):
        if (repo_root / lockfile).is_file():
            lockfiles.add(lockfile)


def _maybe_add_go_topology(
    repo_root: Path,
    entrypoints: list[ProjectEntrypoint],
    anchors: list[ProjectConfigAnchor],
    dependency_manifests: set[str],
    lockfiles: set[str],
) -> None:
    if not (repo_root / "go.mod").is_file():
        return
    dependency_manifests.add("go.mod")
    if (repo_root / "go.sum").is_file():
        lockfiles.add("go.sum")
    anchors.append(
        ProjectConfigAnchor(
            path="go.mod",
            context="go_module",
            language="go",
            ecosystem="go",
            anchor_type="go-module",
            status="present",
        )
    )
    main_go = repo_root / "main.go"
    if main_go.is_file():
        entrypoints.append(
            ProjectEntrypoint(
                path="main.go",
                context="go_main",
                language="go",
                ecosystem="go",
                config_anchor_path="go.mod",
            )
        )
    cmd_root = repo_root / "cmd"
    if cmd_root.is_dir():
        for candidate in sorted(cmd_root.glob("*/main.go"))[:20]:
            entrypoints.append(
                ProjectEntrypoint(
                    path=candidate.relative_to(repo_root).as_posix(),
                    context="go_cmd",
                    language="go",
                    ecosystem="go",
                    config_anchor_path="go.mod",
                )
            )


def _maybe_add_rust_topology(
    repo_root: Path,
    entrypoints: list[ProjectEntrypoint],
    anchors: list[ProjectConfigAnchor],
    dependency_manifests: set[str],
    lockfiles: set[str],
) -> None:
    cargo = repo_root / "Cargo.toml"
    if not cargo.is_file():
        return
    dependency_manifests.add("Cargo.toml")
    if (repo_root / "Cargo.lock").is_file():
        lockfiles.add("Cargo.lock")
    anchors.append(
        ProjectConfigAnchor(
            path="Cargo.toml",
            context="cargo",
            language="rust",
            ecosystem="rust",
            anchor_type="cargo-manifest",
            status="present",
        )
    )
    if (repo_root / "src" / "main.rs").is_file():
        entrypoints.append(
            ProjectEntrypoint(
                path="src/main.rs",
                context="cargo_bin",
                language="rust",
                ecosystem="rust",
                config_anchor_path="Cargo.toml",
            )
        )
    payload = _read_toml(cargo)
    workspace = payload.get("workspace", {}) if isinstance(payload, dict) else {}
    members = workspace.get("members", []) if isinstance(workspace, dict) else []
    if isinstance(members, list):
        for member in members[:20]:
            if not isinstance(member, str):
                continue
            member_manifest = repo_root / member / "Cargo.toml"
            if member_manifest.is_file():
                entrypoints.append(
                    ProjectEntrypoint(
                        path=member_manifest.relative_to(repo_root).as_posix(),
                        context="cargo_workspace_member",
                        language="rust",
                        ecosystem="rust",
                        config_anchor_path="Cargo.toml",
                    )
                )


def _maybe_add_java_kotlin_topology(
    repo_root: Path,
    entrypoints: list[ProjectEntrypoint],
    anchors: list[ProjectConfigAnchor],
    dependency_manifests: set[str],
    lockfiles: set[str],
) -> None:
    has_java = False
    for candidate in ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"):
        if (repo_root / candidate).is_file():
            dependency_manifests.add(candidate)
            anchors.append(
                ProjectConfigAnchor(
                    path=candidate,
                    context="java_build",
                    language="kotlin" if candidate.endswith(".kts") else "java",
                    ecosystem="java",
                    anchor_type="build-manifest",
                    status="present",
                )
            )
            has_java = True
    if not has_java:
        return
    for pattern, language in (
        ("src/main/java/**/*Application.java", "java"),
        ("src/main/java/**/*Main.java", "java"),
        ("src/main/kotlin/**/*Application.kt", "kotlin"),
        ("src/main/kotlin/**/*Main.kt", "kotlin"),
    ):
        for candidate in sorted(repo_root.glob(pattern))[:20]:
            if candidate.is_file():
                anchor_path = "pom.xml" if (repo_root / "pom.xml").is_file() else (
                    "build.gradle.kts" if (repo_root / "build.gradle.kts").is_file() else "build.gradle"
                )
                entrypoints.append(
                    ProjectEntrypoint(
                        path=candidate.relative_to(repo_root).as_posix(),
                        context="jvm_main",
                        language=language,
                        ecosystem="java",
                        config_anchor_path=anchor_path,
                    )
                )
    if (repo_root / ".gradle").is_dir():
        lockfiles.add(".gradle")


def _maybe_add_python_manifests(
    repo_root: Path,
    dependency_manifests: set[str],
    lockfiles: set[str],
) -> None:
    for candidate in (
        "pyproject.toml",
        "requirements.txt",
        "src/requirements.txt",
        "requirements/base.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "requirements/dev.txt",
        "requirements/test.txt",
    ):
        if (repo_root / candidate).is_file():
            dependency_manifests.add(candidate)
    for candidate in ("poetry.lock", "pdm.lock", "uv.lock"):
        if (repo_root / candidate).is_file():
            lockfiles.add(candidate)


def _manifest_hits(repo_root: Path) -> dict[str, bool]:
    return {
        "python": any(
            (repo_root / candidate).is_file()
            for candidate in ("pyproject.toml", "requirements.txt", "src/requirements.txt", "setup.py", "setup.cfg")
        ),
        "node": (repo_root / "package.json").is_file(),
        "go": (repo_root / "go.mod").is_file(),
        "rust": (repo_root / "Cargo.toml").is_file(),
        "java": any(
            (repo_root / candidate).is_file()
            for candidate in ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts")
        ),
    }


def _languages_for_ecosystem(ecosystem: str) -> tuple[str, ...]:
    return {
        "python": ("python",),
        "node": ("javascript", "typescript"),
        "go": ("go",),
        "rust": ("rust",),
        "java": ("java", "kotlin"),
    }.get(ecosystem, tuple())


def _ecosystem_for_language(language: str) -> str:
    return {
        "python": "python",
        "javascript": "node",
        "typescript": "node",
        "go": "go",
        "rust": "rust",
        "java": "java",
        "kotlin": "java",
    }.get(language, "generic")


def _should_skip(path: Path, repo_root: Path) -> bool:
    relative_parts = set(path.relative_to(repo_root).parts)
    return bool(relative_parts & SKIP_PARTS)


def _is_test_path(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or normalized.startswith("src/test/")
        or normalized.startswith("src/test/")
        or normalized.startswith("src/main/") is False and "/test/" in normalized
        or normalized.endswith("_test.py")
        or normalized.endswith(".spec.ts")
        or normalized.endswith(".test.ts")
        or normalized.endswith(".spec.js")
        or normalized.endswith(".test.js")
        or normalized.endswith("_test.go")
        or normalized.endswith("test.rs")
    )


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_like(path: Path) -> dict[str, object]:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dedupe_entrypoints(items: list[ProjectEntrypoint]) -> list[ProjectEntrypoint]:
    deduped: list[ProjectEntrypoint] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.path, item.context, item.ecosystem)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_anchors(items: list[ProjectConfigAnchor]) -> list[ProjectConfigAnchor]:
    deduped: list[ProjectConfigAnchor] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.path, item.context, item.anchor_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
