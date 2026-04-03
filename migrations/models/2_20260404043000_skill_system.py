from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


def _sqlite_upgrade_sql() -> str:
    return """
        ALTER TABLE "agent_sessions" ADD COLUMN "active_skill_version" VARCHAR(128);
        ALTER TABLE "agent_sessions" ADD COLUMN "candidate_skill_version" VARCHAR(128);
        ALTER TABLE "agent_sessions" ADD COLUMN "skill_profile_hash" VARCHAR(128);
        ALTER TABLE "tool_calls" ADD COLUMN "active_skill_version" VARCHAR(128);
        ALTER TABLE "tool_calls" ADD COLUMN "candidate_skill_version" VARCHAR(128);
        ALTER TABLE "tool_calls" ADD COLUMN "skill_profile_hash" VARCHAR(128);
        CREATE TABLE IF NOT EXISTS "skill_pack_records" (
            "id" INTEGER PRIMARY KEY AUTOINCREMENT,
            "repo_root" TEXT NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "name" VARCHAR(128) NOT NULL,
            "version" VARCHAR(128) NOT NULL,
            "description" TEXT NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "source" VARCHAR(32) NOT NULL,
            "system_prompt" TEXT NOT NULL,
            "skill_markdown" TEXT NOT NULL,
            "examples" JSON NOT NULL,
            "policy" JSON NOT NULL,
            "profile_hash" VARCHAR(128) NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_skill_pack_repo_agent_version"
            ON "skill_pack_records" ("repo_root", "agent_kind", "version");
        CREATE TABLE IF NOT EXISTS "skill_binding_records" (
            "id" INTEGER PRIMARY KEY AUTOINCREMENT,
            "repo_root" TEXT NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "active_version" VARCHAR(128) NOT NULL,
            "source" VARCHAR(32) NOT NULL,
            "frozen" BOOL NOT NULL DEFAULT 0,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_skill_binding_repo_agent"
            ON "skill_binding_records" ("repo_root", "agent_kind");
        CREATE TABLE IF NOT EXISTS "skill_candidate_records" (
            "candidate_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "based_on_version" VARCHAR(128) NOT NULL,
            "version" VARCHAR(128) NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "shadow_runs" INT NOT NULL DEFAULT 0,
            "notes" JSON NOT NULL,
            "skill_payload" JSON NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS "skill_evaluation_records" (
            "evaluation_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "run_id" VARCHAR(32) NOT NULL,
            "active_version" VARCHAR(128) NOT NULL,
            "candidate_version" VARCHAR(128),
            "active_score" REAL NOT NULL,
            "candidate_score" REAL,
            "promoted" BOOL NOT NULL DEFAULT 0,
            "reasons" JSON NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_skill_evaluations_run_id"
            ON "skill_evaluation_records" ("run_id");
        CREATE TABLE IF NOT EXISTS "agent_reflection_records" (
            "reflection_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "run_id" VARCHAR(32) NOT NULL,
            "task_id" VARCHAR(64) NOT NULL,
            "session_id" VARCHAR(32) NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "skill_version" VARCHAR(128) NOT NULL,
            "summary" TEXT NOT NULL,
            "metrics" JSON NOT NULL,
            "upgrade_hints" JSON NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_agent_reflections_run_id"
            ON "agent_reflection_records" ("run_id");
        CREATE INDEX IF NOT EXISTS "idx_agent_reflections_task_id"
            ON "agent_reflection_records" ("task_id");
        CREATE INDEX IF NOT EXISTS "idx_agent_reflections_session_id"
            ON "agent_reflection_records" ("session_id");"""


def _postgres_upgrade_sql() -> str:
    return """
        ALTER TABLE "agent_sessions" ADD COLUMN IF NOT EXISTS "active_skill_version" VARCHAR(128);
        ALTER TABLE "agent_sessions" ADD COLUMN IF NOT EXISTS "candidate_skill_version" VARCHAR(128);
        ALTER TABLE "agent_sessions" ADD COLUMN IF NOT EXISTS "skill_profile_hash" VARCHAR(128);
        ALTER TABLE "tool_calls" ADD COLUMN IF NOT EXISTS "active_skill_version" VARCHAR(128);
        ALTER TABLE "tool_calls" ADD COLUMN IF NOT EXISTS "candidate_skill_version" VARCHAR(128);
        ALTER TABLE "tool_calls" ADD COLUMN IF NOT EXISTS "skill_profile_hash" VARCHAR(128);
        CREATE TABLE IF NOT EXISTS "skill_pack_records" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "name" VARCHAR(128) NOT NULL,
            "version" VARCHAR(128) NOT NULL,
            "description" TEXT NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "source" VARCHAR(32) NOT NULL,
            "system_prompt" TEXT NOT NULL,
            "skill_markdown" TEXT NOT NULL,
            "examples" JSONB NOT NULL,
            "policy" JSONB NOT NULL,
            "profile_hash" VARCHAR(128) NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_skill_pack_repo_agent_version"
            ON "skill_pack_records" ("repo_root", "agent_kind", "version");
        CREATE TABLE IF NOT EXISTS "skill_binding_records" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "active_version" VARCHAR(128) NOT NULL,
            "source" VARCHAR(32) NOT NULL,
            "frozen" BOOL NOT NULL DEFAULT FALSE,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_skill_binding_repo_agent"
            ON "skill_binding_records" ("repo_root", "agent_kind");
        CREATE TABLE IF NOT EXISTS "skill_candidate_records" (
            "candidate_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "based_on_version" VARCHAR(128) NOT NULL,
            "version" VARCHAR(128) NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "shadow_runs" INT NOT NULL DEFAULT 0,
            "notes" JSONB NOT NULL,
            "skill_payload" JSONB NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS "skill_evaluation_records" (
            "evaluation_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "run_id" VARCHAR(32) NOT NULL,
            "active_version" VARCHAR(128) NOT NULL,
            "candidate_version" VARCHAR(128),
            "active_score" DOUBLE PRECISION NOT NULL,
            "candidate_score" DOUBLE PRECISION,
            "promoted" BOOL NOT NULL DEFAULT FALSE,
            "reasons" JSONB NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_skill_evaluations_run_id"
            ON "skill_evaluation_records" ("run_id");
        CREATE TABLE IF NOT EXISTS "agent_reflection_records" (
            "reflection_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "run_id" VARCHAR(32) NOT NULL,
            "task_id" VARCHAR(64) NOT NULL,
            "session_id" VARCHAR(32) NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "skill_version" VARCHAR(128) NOT NULL,
            "summary" TEXT NOT NULL,
            "metrics" JSONB NOT NULL,
            "upgrade_hints" JSONB NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_agent_reflections_run_id"
            ON "agent_reflection_records" ("run_id");
        CREATE INDEX IF NOT EXISTS "idx_agent_reflections_task_id"
            ON "agent_reflection_records" ("task_id");
        CREATE INDEX IF NOT EXISTS "idx_agent_reflections_session_id"
            ON "agent_reflection_records" ("session_id");"""


async def upgrade(db: BaseDBAsyncClient) -> str:
    if db.capabilities.dialect == "sqlite":
        return _sqlite_upgrade_sql()
    return _postgres_upgrade_sql()


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "agent_reflection_records";
        DROP TABLE IF EXISTS "skill_evaluation_records";
        DROP TABLE IF EXISTS "skill_candidate_records";
        DROP TABLE IF EXISTS "skill_binding_records";
        DROP TABLE IF EXISTS "skill_pack_records";"""


MODELS_STATE = "eJyrVsrNT0nNKVayUqhWKs7OzMmJL0hMzo4vSk3OL0qBCJckJuWkAlnY5GtrawEt2hhv"
