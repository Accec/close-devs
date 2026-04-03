from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "runs" (
            "run_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "workflow_name" VARCHAR(128) NOT NULL,
            "repo_root" TEXT NOT NULL,
            "started_at" TIMESTAMPTZ NOT NULL,
            "finished_at" TIMESTAMPTZ,
            "status" VARCHAR(32) NOT NULL,
            "summary" TEXT,
            "report_dir" TEXT
        );
        CREATE TABLE IF NOT EXISTS "snapshots" (
            "run_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "scanned_at" TIMESTAMPTZ NOT NULL,
            "revision" TEXT,
            "file_hashes" JSONB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS "tasks" (
            "task_id" VARCHAR(64) NOT NULL PRIMARY KEY,
            "run_id" VARCHAR(32) NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "task_type" VARCHAR(32) NOT NULL,
            "targets" JSONB NOT NULL,
            "payload_summary" JSONB NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "summary" TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_tasks_run_id" ON "tasks" ("run_id");
        CREATE TABLE IF NOT EXISTS "findings" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "run_id" VARCHAR(32) NOT NULL,
            "task_id" VARCHAR(64) NOT NULL,
            "agent_kind" VARCHAR(32) NOT NULL,
            "severity" VARCHAR(32) NOT NULL,
            "rule_id" VARCHAR(255) NOT NULL,
            "category" VARCHAR(64) NOT NULL,
            "path" TEXT,
            "line" INT,
            "symbol" TEXT,
            "message" TEXT NOT NULL,
            "fingerprint" TEXT NOT NULL,
            "evidence" JSONB NOT NULL,
            "state" VARCHAR(32) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_findings_run_id" ON "findings" ("run_id");
        CREATE INDEX IF NOT EXISTS "idx_findings_task_id" ON "findings" ("task_id");
        CREATE TABLE IF NOT EXISTS "patches" (
            "run_id" VARCHAR(32) NOT NULL PRIMARY KEY,
            "summary" TEXT NOT NULL,
            "rationale" TEXT NOT NULL,
            "touched_files" JSONB NOT NULL,
            "diff_text" TEXT NOT NULL,
            "suggestions" JSONB NOT NULL,
            "applied" BOOL NOT NULL,
            "file_patches" JSONB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS "issue_catalog" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "repo_root" TEXT NOT NULL,
            "fingerprint" TEXT NOT NULL,
            "rule_id" VARCHAR(255) NOT NULL,
            "path" TEXT,
            "message" TEXT NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "first_seen_run" VARCHAR(32) NOT NULL,
            "last_seen_run" VARCHAR(32) NOT NULL,
            "occurrences" INT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_issue_catalog_repo_fingerprint"
            ON "issue_catalog" ("repo_root", "fingerprint");"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "issue_catalog";
        DROP TABLE IF EXISTS "patches";
        DROP TABLE IF EXISTS "findings";
        DROP TABLE IF EXISTS "tasks";
        DROP TABLE IF EXISTS "snapshots";
        DROP TABLE IF EXISTS "runs";"""


MODELS_STATE = (
    "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"
)
