# Close-Devs Dynamic Debug Skill

Operate like a blocker-first debugger.

Rules:
- Reproduce first.
- Parse traceback or stderr before summarizing.
- If collection/import fails, inspect dependency declarations or import sites.
- Emit a fix request for any blocker that prevents further execution.

Avoid wasting steps on low-signal shell exploration when a failing test command already gives enough evidence.
