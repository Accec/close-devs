from __future__ import annotations

from datetime import datetime, timedelta, timezone


class Scheduler:
    def should_run(
        self,
        last_finished_at: str | None,
        scan_interval_minutes: int,
        now: datetime | None = None,
    ) -> bool:
        if last_finished_at is None:
            return True
        current_time = now or datetime.now(timezone.utc)
        last_run = datetime.fromisoformat(last_finished_at)
        return current_time - last_run >= timedelta(minutes=scan_interval_minutes)

