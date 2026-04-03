from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


def _sqlite_upgrade_sql() -> str:
    return """
        CREATE TABLE IF NOT EXISTS "agent_sessions" (
            "session_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "run_id" VARCHAR(32) NOT NULL,
            "task_id" VARCHAR(64) NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "task_type" VARCHAR(32) NOT NULL,
            "working_repo_root" TEXT NOT NULL,
            "objective" TEXT NOT NULL,
            "started_at" TIMESTAMPTZ NOT NULL,
            "finished_at" TIMESTAMPTZ,
            "completion_reason" VARCHAR(64),
            "summary" TEXT,
            "step_count" INT NOT NULL DEFAULT 0,
            "tool_call_count" INT NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS "idx_agent_sessions_run_id" ON "agent_sessions" ("run_id");
        CREATE INDEX IF NOT EXISTS "idx_agent_sessions_task_id" ON "agent_sessions" ("task_id");
        CREATE TABLE IF NOT EXISTS "agent_steps" (
            "id" INTEGER PRIMARY KEY AUTOINCREMENT,
            "session_id" VARCHAR(32) NOT NULL,
            "step_index" INT NOT NULL,
            "decision_summary" TEXT NOT NULL,
            "action_type" VARCHAR(32) NOT NULL,
            "tool_name" VARCHAR(128),
            "tool_input" JSON NOT NULL,
            "final_response" JSON NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_agent_steps_session_id" ON "agent_steps" ("session_id");
        CREATE TABLE IF NOT EXISTS "agent_handoffs" (
            "id" INTEGER PRIMARY KEY AUTOINCREMENT,
            "session_id" VARCHAR(32) NOT NULL,
            "run_id" VARCHAR(32) NOT NULL,
            "task_id" VARCHAR(64) NOT NULL,
            "source_agent" VARCHAR(32) NOT NULL,
            "title" TEXT NOT NULL,
            "description" TEXT NOT NULL,
            "recommended_change" TEXT NOT NULL,
            "severity" VARCHAR(32) NOT NULL,
            "affected_files" JSON NOT NULL,
            "evidence" JSON NOT NULL,
            "metadata" JSON NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_agent_handoffs_session_id" ON "agent_handoffs" ("session_id");
        CREATE INDEX IF NOT EXISTS "idx_agent_handoffs_run_id" ON "agent_handoffs" ("run_id");
        CREATE TABLE IF NOT EXISTS "tool_calls" (
            "id" INTEGER PRIMARY KEY AUTOINCREMENT,
            "session_id" VARCHAR(32) NOT NULL,
            "step_index" INT NOT NULL,
            "tool_name" VARCHAR(128) NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "tool_input" JSON NOT NULL,
            "output" JSON NOT NULL,
            "summary" TEXT,
            "error" TEXT,
            "started_at" TIMESTAMPTZ NOT NULL,
            "finished_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_tool_calls_session_id" ON "tool_calls" ("session_id");"""


def _postgres_upgrade_sql() -> str:
    return """
        CREATE TABLE IF NOT EXISTS "agent_sessions" (
            "session_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "run_id" VARCHAR(32) NOT NULL,
            "task_id" VARCHAR(64) NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "task_type" VARCHAR(32) NOT NULL,
            "working_repo_root" TEXT NOT NULL,
            "objective" TEXT NOT NULL,
            "started_at" TIMESTAMPTZ NOT NULL,
            "finished_at" TIMESTAMPTZ,
            "completion_reason" VARCHAR(64),
            "summary" TEXT,
            "step_count" INT NOT NULL DEFAULT 0,
            "tool_call_count" INT NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS "idx_agent_sessions_run_id" ON "agent_sessions" ("run_id");
        CREATE INDEX IF NOT EXISTS "idx_agent_sessions_task_id" ON "agent_sessions" ("task_id");
        CREATE TABLE IF NOT EXISTS "agent_steps" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "session_id" VARCHAR(32) NOT NULL,
            "step_index" INT NOT NULL,
            "decision_summary" TEXT NOT NULL,
            "action_type" VARCHAR(32) NOT NULL,
            "tool_name" VARCHAR(128),
            "tool_input" JSONB NOT NULL,
            "final_response" JSONB NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_agent_steps_session_id" ON "agent_steps" ("session_id");
        CREATE TABLE IF NOT EXISTS "agent_handoffs" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "session_id" VARCHAR(32) NOT NULL,
            "run_id" VARCHAR(32) NOT NULL,
            "task_id" VARCHAR(64) NOT NULL,
            "source_agent" VARCHAR(32) NOT NULL,
            "title" TEXT NOT NULL,
            "description" TEXT NOT NULL,
            "recommended_change" TEXT NOT NULL,
            "severity" VARCHAR(32) NOT NULL,
            "affected_files" JSONB NOT NULL,
            "evidence" JSONB NOT NULL,
            "metadata" JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_agent_handoffs_session_id" ON "agent_handoffs" ("session_id");
        CREATE INDEX IF NOT EXISTS "idx_agent_handoffs_run_id" ON "agent_handoffs" ("run_id");
        CREATE TABLE IF NOT EXISTS "tool_calls" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "session_id" VARCHAR(32) NOT NULL,
            "step_index" INT NOT NULL,
            "tool_name" VARCHAR(128) NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "tool_input" JSONB NOT NULL,
            "output" JSONB NOT NULL,
            "summary" TEXT,
            "error" TEXT,
            "started_at" TIMESTAMPTZ NOT NULL,
            "finished_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_tool_calls_session_id" ON "tool_calls" ("session_id");"""


async def upgrade(db: BaseDBAsyncClient) -> str:
    if db.capabilities.dialect == "sqlite":
        return _sqlite_upgrade_sql()
    return _postgres_upgrade_sql()


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "tool_calls";
        DROP TABLE IF EXISTS "agent_handoffs";
        DROP TABLE IF EXISTS "agent_steps";
        DROP TABLE IF EXISTS "agent_sessions";"""


MODELS_STATE = (
    "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"
    "njkzyUAgNTGlJludXn78BJysYYo="
)
