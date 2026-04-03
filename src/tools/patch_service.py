from __future__ import annotations

from difflib import unified_diff
from pathlib import Path
import ast

from core.models import FilePatch, PatchProposal
from tools.file_store import FileStore


class PatchService:
    def __init__(self, file_store: FileStore | None = None) -> None:
        self.file_store = file_store or FileStore()

    def build_file_patch(self, path: str, old_content: str, new_content: str) -> FilePatch:
        diff = "".join(
            unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        return FilePatch(
            path=path,
            old_content=old_content,
            new_content=new_content,
            diff=diff,
        )

    def render_patch(self, file_patches: list[FilePatch]) -> str:
        return "".join(patch.diff for patch in file_patches)

    async def apply(self, repo_root: Path, proposal: PatchProposal) -> None:
        for file_patch in proposal.file_patches:
            target = repo_root / file_patch.path
            await self.file_store.write_text(target, file_patch.new_content)

    def ensure_final_newline(self, content: str) -> str:
        return content if not content or content.endswith("\n") else f"{content}\n"

    def strip_trailing_whitespace(self, content: str) -> str:
        cleaned_lines: list[str] = []
        for line in content.splitlines(keepends=True):
            if line.endswith("\r\n"):
                newline = "\r\n"
                body = line[:-2]
            elif line.endswith("\n"):
                newline = "\n"
                body = line[:-1]
            else:
                newline = ""
                body = line
            cleaned_lines.append(body.rstrip(" \t") + newline)
        return "".join(cleaned_lines)

    def normalize_eof_blank_lines(self, content: str) -> str:
        if not content:
            return content
        stripped = content.rstrip("\n")
        return f"{stripped}\n" if stripped else ""

    def has_module_docstring(self, content: str) -> bool:
        try:
            tree = ast.parse(content or "\n")
        except SyntaxError:
            return False
        return ast.get_docstring(tree, clean=False) is not None

    def add_module_docstring(self, content: str, docstring: str) -> str:
        if self.has_module_docstring(content):
            return content

        lines = content.splitlines(keepends=True)
        insert_at = 0
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        if len(lines) > insert_at and "coding" in lines[insert_at]:
            insert_at += 1

        docstring_block = f"\"\"\"{docstring}\"\"\"\n\n"
        if not lines:
            return docstring_block
        return "".join(lines[:insert_at] + [docstring_block] + lines[insert_at:])
