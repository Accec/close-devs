# Close-Devs Static Review Skill

Focus order:
1. Correctness and runtime contracts
2. Architecture/bootstrap/initialization risks
3. Dependency/import hazards
4. Security-relevant exception handling
5. Cosmetic lint only if no stronger issue exists

Start from `session.payload.static_context` before choosing tools:
- use `startup_topology` and config anchors to understand entrypoints
- use `top_targets` and `high_signal_targets` to prioritize inspection
- use `baseline_static_digest` to avoid getting trapped by docstring-only noise
- use `related_files` and import adjacency to expand one hop when the first file suggests a cross-file issue

When deterministic tooling is noisy, inspect at least one central module and one neighboring dependency edge before finalizing.

`run_static_review` is a verification and drill-down tool. Do not treat it as the only first-round source of repository context.

When you identify a medium or higher severity issue, produce a fix request with:
- affected files
- evidence
- concrete recommended change

Do not let docstring-only output dominate the final review when stronger risks exist.
