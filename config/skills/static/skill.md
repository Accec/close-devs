# Close-Devs Static Review Skill

Focus order:
1. Correctness and runtime contracts
2. Architecture/bootstrap/initialization risks
3. Dependency/import hazards
4. Security-relevant exception handling
5. Cosmetic lint only if no stronger issue exists

When deterministic tooling is noisy, inspect at least one central module and one neighboring dependency edge before finalizing.

When you identify a medium or higher severity issue, produce a fix request with:
- affected files
- evidence
- concrete recommended change

Do not let docstring-only output dominate the final review when stronger risks exist.
