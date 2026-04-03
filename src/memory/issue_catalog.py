from __future__ import annotations

from core.models import Finding
from memory.state_store import StateStore


class IssueCatalog:
    def __init__(self, state_store: StateStore) -> None:
        self.state_store = state_store

    async def record_findings(self, repo_root: str, run_id: str, findings: list[Finding]) -> None:
        for finding in findings:
            await self.state_store.upsert_issue(repo_root, finding, run_id, status="open")

    async def reconcile(
        self,
        repo_root: str,
        run_id: str,
        original: list[Finding],
        validated: list[Finding],
    ) -> dict[str, list[str]]:
        original_fingerprints = {finding.fingerprint for finding in original}
        validated_fingerprints = {finding.fingerprint for finding in validated}

        resolved = sorted(original_fingerprints - validated_fingerprints)
        unresolved = sorted(original_fingerprints & validated_fingerprints)
        regressed = sorted(validated_fingerprints - original_fingerprints)

        for fingerprint in resolved:
            await self.state_store.set_issue_status(repo_root, fingerprint, "resolved", run_id)
        for fingerprint in unresolved:
            await self.state_store.set_issue_status(repo_root, fingerprint, "open", run_id)
        for finding in validated:
            if finding.fingerprint in regressed:
                await self.state_store.upsert_issue(repo_root, finding, run_id, status="regressed")

        return {
            "resolved": resolved,
            "unresolved": unresolved,
            "regressed": regressed,
        }
